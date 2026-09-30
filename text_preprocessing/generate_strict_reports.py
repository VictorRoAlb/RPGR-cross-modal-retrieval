#!/usr/bin/env python3
"""
filter_reports_no_diagnosis.py
================================
Anade a tcga_reports_cleaned.csv una columna nueva, slide_reports_no_diagnosis_strict,
con una reescritura de slide_reports (el campo usado para el encoder de texto en el
retrieval I2T/T2I) que describe UNICAMENTE morfologia histologica -- sin diagnostico,
sin grado/estadio, sin inmunohistoquimica/biomarcadores, sin el organo usado como
introduccion ("The slide from the X shows..."), y sin ningun otro "keyword" que
delate la clase. Usa el mismo proceso ya validado para ASSIST/SICAP: openai (gpt-4.1)
con Structured Outputs (json_schema, strict) para que el modelo no se invente nada
fuera del formato esperado, mas una validacion automatica por regex contra fuga de
diagnostico y reintento con prompt mas estricto si se detecta.

Por que hace falta: slide_reports regala el diagnostico literal y el organo, que
son las mismas etiquetas usadas para clasificacion (OncotreeCode/cancer_type_detailed),
inflando artificialmente las metricas I2T/T2I por fuga de keywords en vez de por
alineacion real imagen-texto (ver results/eval/excel/tcga_results_summary.xlsx).

Nota sobre la fuente: se usa solo slide_reports, no case_reports. slide_reports es
corto (max ~700 caracteres) y muchas veces mayoritariamente diagnostico -- para esas
filas el resultado sera legitimamente corto una vez quitado el diagnostico. Eso es
una caracteristica real de los datos de origen, no un fallo del script: nunca se debe
inventar morfologia que no este en el texto original.

Validacion anti-fuga: ademas de una lista de patrones prohibidos universales
(grado/estadio, IHQ, frases interpretativas, margenes/extension de la muestra --
validos para cualquier tipo de cancer), se comprueban terminos DINAMICOS por fila,
derivados de las propias columnas de metadatos de esa fila (cancer_type_detailed,
site_of_resection_or_biopsy, OncotreeCode) -- necesario porque TCGA es pan-cancer
(32 tipos, ver project_id) y no se puede enumerar a mano cada organo/entidad.

Es un bucle sincrono (no Batch API) fila a fila, con checkpoint periodico a disco y
reanudable: si se relanza el script, las filas que ya tienen texto en la columna
nueva se saltan salvo --overwrite.

Uso tipico:
  # 1) piloto pequeno, estratificado por tipo de cancer, para revisar calidad
  #    (nunca toca el CSV maestro; vuelca a un CSV + JSON de revision aparte)
  python3 filter_reports_no_diagnosis.py --pilot 40

  # -> revisar el CSV/JSON de revision (incluye que se quito de cada fila y si
  #    sigue habiendo posible fuga tras los reintentos) antes de seguir

  # 2) tanda completa (cohorte strict-common de 6.727 slides por defecto):
  #    actualiza tcga_reports_cleaned.csv (columna nueva, escritura atomica) y
  #    escribe un JSON de auditoria junto a el con que se quito/cambio por fila
  python3 filter_reports_no_diagnosis.py

  # si se corta a medias, relanzar el mismo comando reanuda donde se quedo
  python3 filter_reports_no_diagnosis.py

Requiere: pip install openai pandas tqdm
Autenticacion: variable de entorno OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI, RateLimitError, APIError, APITimeoutError
except ImportError:
    sys.exit("Falta el SDK de OpenAI. Instala con: pip install openai")


# No default paths shipped with this release -- point --csv and
# --strict-common-ids at your own master reports CSV and cohort id list
# (see docs/DATA_FORMAT.md). Originally defaulted to this project's own
# internal paths; that default was removed here, nothing else changed.
DEFAULT_CSV = None
DEFAULT_STRICT_COMMON_IDS = None
SOURCE_COLUMN = "slide_reports"
NEW_COLUMN = "slide_reports_no_diagnosis_strict"
CSV_SEP = ";"
# NOTA: "OncotreeCode" se quito de aqui (encontrado en la auditoria final de la
# tanda completa). Es un codigo corto (p.ej. "ASTR" para Astrocytoma, "ODG"
# para Oligodendroglioma) que no aporta cobertura nueva -- cancer_type_detailed
# ya trae el mismo dato en texto completo ("Astrocytoma") -- pero al ser corto,
# el matching dinamico con comodin (\bTERM\w*\b) lo convertia en un PREFIJO que
# bloqueaba palabras legitimas: "ASTR" bloqueaba "astrocytic"/"astrocytic
# features" (un descriptor de linaje celular real, no el nombre de la entidad)
# en filas de LGG. cancer_type_detailed solo, al venir en palabras completas,
# no tiene este problema.
METADATA_COLUMNS_FOR_DYNAMIC_TERMS = ["cancer_type_detailed", "site_of_resection_or_biopsy"]


# ---------------- TEXT UTILS ----------------

def one_paragraph(s) -> str:
    s = str(s or "").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = s.replace("\n", " ")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def ensure_period(s: str) -> str:
    s = (s or "").strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s


def parse_retry_seconds(msg: str) -> Optional[int]:
    m = re.search(r"try again in (\d+)s", msg, re.I)
    return int(m.group(1)) if m else None


def has_useful_text(s) -> bool:
    if pd.isna(s):
        return False
    return len(one_paragraph(str(s))) >= 15


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep=CSV_SEP, index=False)
    tmp.replace(path)


def safe_write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


# ---------------- ANTI-LEAKAGE: patrones universales (pan-cancer) ----------------

STATIC_FORBIDDEN_PATTERNS = [
    # nombres de entidad diagnostica (raices genericas, cubren la mayoria de
    # los 32 tipos de proyecto sin tener que enumerar cada combinacion)
    r"\bcarcinoma\b", r"\badenocarcinoma\b", r"\bsarcoma\b", r"\blymphoma\b",
    r"\bmelanoma\b", r"\bglioma\b", r"\bglioblastoma\b", r"\bastrocytoma\b",
    r"\boligodendroglioma\b", r"\bependymoma\b", r"\bmedulloblastoma\b",
    r"\bmesothelioma\b", r"\bthymoma\b", r"\bseminoma\b", r"\bchordoma\b",
    r"\bmeningioma\b", r"\bschwannoma\b", r"\bneurofibroma\b",
    r"\bpheochromocytoma\b", r"\bparaganglioma\b", r"\bteratoma\b",
    r"\bgerm cell (tumor|neoplasia)\b",
    # NOTA: "ductal"/"lobular" NO estan aqui a proposito (se quitaron de esta
    # lista estatica tras encontrarlo en un piloto real). Son subtipo
    # diagnostico SOLO en BRCA ("invasive ductal/lobular carcinoma"), pero en
    # el resto de tipos de cancer (y a veces incluso dentro de BRCA) son
    # vocabulario de arquitectura legitimo ("tubulo-lobular features",
    # "ductal-pattern growth" -- el propio prompt invita a usar este ultimo).
    # Bloquearlos en bloque hacia fallar casos correctos: un tumor DUCTAL
    # descrito con un patron "tubulo-lobular" es una observacion real, no una
    # fuga del subtipo lobular. En vez de una regla estatica universal, se
    # dejan solo en dynamic_forbidden_terms(): si la propia fila es, p.ej.,
    # "Breast Invasive Ductal Carcinoma" segun cancer_type_detailed, "ductal"
    # se prohibe SOLO para esa fila (es su etiqueta real); si la fila es de
    # otro tipo de cancer, o es lobular en vez de ductal, la palabra queda
    # libre para usarse como arquitectura.
    # NOTA: "neuroendocrine", "melanocytic", "anaplastic" y similares NO estan
    # aqui a proposito -- son descriptores de tipo/linaje celular observables
    # (que celulas se ven), no la etiqueta final de la entidad. Segun Victor:
    # se conservan aunque den pista de la clase, mientras describan lo que
    # realmente se ve. Ver prompt (distincion etiqueta vs. descripcion).

    # metastasis / origen
    r"\bmetastasis\b", r"\bmetastatic\b", r"\bsecondary\b",
    r"\bprimary tumor\b", r"\bsite of origin\b",

    # lenguaje interpretativo/conclusivo
    r"\bcompatible with\b", r"\bconsistent with\b", r"\bsuggestive of\b",
    r"\bin keeping with\b", r"\bfavor(s|ing)?\b", r"\bdiagnostic of\b",

    # meta-comentario sobre el propio proceso de reescritura/filtrado (el modelo
    # hablando de sus instrucciones en vez de describir el tejido -- visto en
    # produccion real, ver conversacion)
    r"\bdiagnostic leakage\b", r"\bmust be omitted\b", r"\bhas been (removed|omitted)\b",
    r"\bthese instructions\b", r"\bto (avoid|prevent) (revealing|disclosing|leaking)\b",

    # grado / estadificacion como ETIQUETA FORMAL de clasificacion (numero,
    # sistema con nombre, o el adjetivo "high/low-grade" usado como clase) --
    # OJO: "well/moderately/poorly differentiated" y "undifferentiated" NO
    # estan aqui a proposito. Segun feedback de Victor: la diferenciacion es
    # una propiedad morfologica real y observable (organizacion celular,
    # maduracion), distinta de un numero de grado formal -- se permite
    # conservarla reformulada como observacion (ver prompt), no se prohibe
    # en bloque. Ver conversacion.
    r"\bwho grade\b", r"\bgleason\b", r"\bgrade\s*[ivx1-4]+\b",
    r"\bhigh[\s-]grade\b", r"\blow[\s-]grade\b",

    # estadificacion
    r"\bp[tnm][0-4x]\b", r"\bstage [i1]+[iv]*\b", r"\btnm\b", r"\bconfined to\b",

    # margenes / extension de la muestra
    r"\bmargin(s)?\b", r"\bresection margin\b", r"\bsurgical margin\b",
    r"\bentire specimen\b", r"\bwhole specimen\b", r"\bentire sample\b",

    # IHQ / biomarcadores / molecular
    r"\bimmunohistochem\w*\b", r"\bmolecular\b", r"\bmutation\b",
    r"\bamplification\b", r"\btranslocation\b", r"\breceptor status\b",
    r"\bher2\b", r"\ber\b(?!\w)", r"\bpr\b(?!\w)", r"\bki[\s-]?67\b", r"\bmib[\s-]?1\b",
    r"\bpten\b", r"\begfr\b", r"\bp53\b", r"\bp63\b", r"\bvimentin\b",
    r"\bck\s*7\b", r"\bck\s*20\b", r"\bttf[\s-]?1\b", r"\bcdx2\b", r"\bgata3\b",
    r"\bpax8\b", r"\bs100\b", r"\bmelan[\s-]?a\b", r"\bcd\d{1,3}\b",
    r"\bsynaptophysin\b", r"\bchromogranin\b", r"\bdesmin\b", r"\bcalretinin\b",
    r"\bhbme1?\b",

    # descripcion MACROSCOPICA/gruesa de la pieza -- estos textos acompanan
    # imagenes de microscopio (WSI, seccion fina tenida), no una foto de la
    # pieza quirurgica entera. Cualquier dato de tamano/consistencia/aspecto
    # de la pieza al ojo o al tacto no se puede observar en una imagen
    # microscopica, asi que se prohibe aunque no sea "diagnostico" -- es
    # simplemente el tipo de dato equivocado para este uso. Confirmado con
    # Victor tras revisar el piloto: ~el 50% de las filas del CSV original
    # tienen una medida en cm, asi que este patron es de alto impacto.
    r"\b\d+(\.\d+)?\s*(cm|mm)\b",
    r"\bfleshy\b", r"\brubbery\b", r"\bfriable\b", r"\bspongy\b", r"\bgritty\b",
    r"\bfirm\b", r"\bcut surface(s)?\b", r"\bgross\b",
]

# palabras genericas de metadatos que son vocabulario morfologico legitimo
# (no delatan nada) -- se excluyen de la lista dinamica por fila. Incluye
# adjetivos de citologia/arquitectura que a veces forman parte del NOMBRE de
# la entidad (p.ej. "High-Grade Spindle Cell Sarcoma" en cancer_type_detailed)
# pero que tambien son vocabulario morfologico legitimo fuera de ese nombre
# (p.ej. "highly infiltrative growth", "pleomorphic nuclei") -- visto en
# produccion real: "high" bloqueaba "highly infiltrative" sin motivo real.
DYNAMIC_TERM_STOPWORDS = {
    "gland", "glands", "tissue", "site", "type", "cell", "cells", "duct",
    "ducts", "nos", "other", "unclassified", "specified", "otherwise", "not",
    "and", "with", "invasive",
    "high", "low", "grade", "spindle", "pleomorphic", "pleomorphism",
    "clear", "giant", "small", "large",
    # encontrados en la auditoria de la tanda completa: igual que
    # "ductal"/"lobular" (ver STATIC_FORBIDDEN_PATTERNS), estas son palabras
    # de arquitectura/patron histologico legitimas y universales que a veces
    # tambien forman parte del NOMBRE del subtipo de la fila (p.ej. "Papillary
    # Serous Carcinoma" en UCEC) -- bloquearlas en bloque impedia describir la
    # arquitectura papilar/serosa/mucinosa real que el propio prompt anima a
    # conservar. "soft"/"subcutaneous" bloqueaban tambien la profundidad de
    # invasion real en SARC ("invasion into deep soft tissue"), no solo el
    # organo de origen.
    "papillary", "serous", "mucinous", "soft", "subcutaneous",
    # "malignant" ya se habia quitado de STATIC_FORBIDDEN_PATTERNS (TCGA es
    # 100% tumor, no discrimina nada) pero se colaba otra vez via dinamico
    # cuando la propia cancer_type_detailed la contenia literalmente (p.ej.
    # "Malignant Peripheral Nerve Sheath Tumor"). "nerve" es el nombre de una
    # estructura anatomica real de la que el tumor puede surgir/a la que
    # puede invadir -- describirlo es una observacion legitima (ver KEEP:
    # "infiltration into a named adjacent structure"), no una fuga, aunque
    # tambien aparezca en el nombre de esta entidad concreta.
    "malignant", "nerve",
}


def dynamic_forbidden_terms(row: pd.Series) -> set[str]:
    """Terminos prohibidos derivados del propio metadato de la fila (organo,
    tipo de cancer detallado, codigo Oncotree) -- necesario porque en TCGA
    cambian por fila (32 tipos), a diferencia de un dataset de un solo
    dominio como ASSIST."""
    terms: set[str] = set()
    for col in METADATA_COLUMNS_FOR_DYNAMIC_TERMS:
        val = str(row.get(col, "") or "")
        for tok in re.split(r"[^A-Za-z]+", val):
            tok_l = tok.lower()
            if len(tok_l) >= 4 and tok_l not in DYNAMIC_TERM_STOPWORDS:
                terms.add(tok_l)
    return terms


def has_forbidden_leakage(text: str, dynamic_terms: set[str]) -> list[str]:
    """Devuelve la lista de patrones/terminos que han saltado (vacia si el
    texto esta limpio) -- se usa tanto para decidir si hay que reintentar
    como para el JSON de auditoria (que se ha quitado / que sigue sucio)."""
    text = text or ""
    matched = [p for p in STATIC_FORBIDDEN_PATTERNS if re.search(p, text, re.I)]
    for term in dynamic_terms:
        if re.search(rf"\b{re.escape(term)}\w*\b", text, re.I):
            matched.append(term)
    return matched


# ---------------- PROMPT ----------------

BASE_CONTEXT_NO_DIAGNOSIS = """
You are producing strict morphology-only histopathology summaries for training and benchmarking
large foundation models in computational pathology (image-text retrieval).

These summaries are specifically designed to REMOVE DIAGNOSTIC LABEL LEAKAGE from the text, so
that a retrieval model cannot match text to image just by reading the diagnosis, grade, stage, or
organ name -- it must learn from genuine visual/morphological description instead.

The output must be:
- strictly grounded in the provided source text -- NEVER invent or add morphological detail that
  is not present in the source
- focused only on microscopic, cellular, cytologic, and architectural information
- fluent, natural, and publication-grade -- readable and coherent, not telegraphic
- concise but complete: keep every genuine morphological detail present in the source, do not
  drop real morphology just to be short
- one single paragraph, free of hallucinations
"""

PROMPT_NO_DIAGNOSIS_STRICT = BASE_CONTEXT_NO_DIAGNOSIS + """

THE CORE PRINCIPLE -- read this first, it governs every rule below:
Remove only the FINAL LABEL, never the underlying visual observation. Strip the diagnostic
entity name, the formal grade/stage classification, and lab test results -- but if the source
describes something a pathologist can actually SEE on the slide, keep it, even if that
observation happens to point toward a particular diagnosis. Losing real histological content is
just as much a failure as leaking the diagnosis label -- the goal is maximum genuine visual
information with zero formal diagnostic labels, not a maximally scrubbed, generic-sounding text.
Concretely: cell-type / lineage / appearance words describe what is actually visible and should be
KEPT even though they hint at the diagnosis -- that hint is an unavoidable side effect of
describing real tissue, not something to launder away -- BUT only when the source independently
describes that appearance as an observation. If the source gives you nothing beyond the entity
name itself, mechanically back-translating the entity name's root into an "-oid"/"-like"/
descriptive form (e.g. turning "squamous cell carcinoma" into "squamoid morphology", or
"adenocarcinoma" into "glandular structures", with no other supporting detail in the source) is
NOT a genuine observation -- it is the diagnosis in a thin disguise, and is just as much a failure
as leaking the entity name directly. See "DO NOT TRANSLATE THE ENTITY NAME INTO ITS OWN
DEFINITION" below for the full rule and worked examples. What must go is the entity NOUN that
names the tumor type itself ("melanoma", "carcinoma", "sarcoma", "glioma", "oligodendroglioma",
etc.) and formal classification labels (grade numbers, staging codes, named scoring systems).

CRITICAL RULES:
- DO NOT mention the final diagnosis or the entity/tumor-type NAME itself (e.g. "carcinoma",
  "adenocarcinoma", "sarcoma", "melanoma", "glioma", "oligodendroglioma", "lymphoma", or any
  other disease/entity noun that names the specific tumor type).
- DO mention cell-type, lineage, or appearance descriptors when the source independently
  describes what the cells actually look like or resemble (beyond just naming the entity), even
  if that description points toward a diagnosis: e.g. "melanocytic cells" is fine when the source
  itself describes an atypical melanocytic proliferation, not merely because the diagnosis is
  "melanoma"; "neuroendocrine-appearing cells" is fine when the source describes the cells that
  way, not merely because the entity name implies it. The line: "oligodendroglioma" is the entity
  name (remove it); "cells with oligodendroglial-like morphology" is only a legitimate description
  of appearance (keep it) if the source gives some independent basis for it beyond the bare entity
  name -- otherwise it is a disguised restatement of the diagnosis, not an observation (see "DO
  NOT TRANSLATE THE ENTITY NAME INTO ITS OWN DEFINITION" below).
- DO NOT mention the organ or anatomical site of origin (drop any "The slide from the X shows..."
  framing entirely -- do not restate the organ anywhere in your output).
- Avoid bare conclusory words like "malignant tumor" or "benign" with nothing else attached --
  if that is genuinely all there is, see the INSUFFICIENT-SOURCE RULE below instead of padding it.
- DO NOT include interpretive conclusions such as "compatible with", "consistent with",
  "suggestive of", "in keeping with", "favors", "diagnostic of".
- DO NOT talk about this task, these instructions, or the redaction/filtering process itself.
  Never write phrases like "terminology that must be omitted", "information has been removed",
  or "to prevent diagnostic leakage" -- these are meta-commentary about the process, not tissue
  description, and are just as much a failure as leaking the diagnosis itself. Output ONLY the
  tissue description, nothing about how or why it was produced.

STRICT ANTI-LEAKAGE FILTER (formal labels and lab results -- these must go):
- DO NOT mention formal grading/staging as a LABEL: Gleason score, "WHO grade N", a bare grade
  number ("Grade 3"), "high-grade"/"low-grade" used as a classification tag, TNM/pT/pN/pM codes,
  named stage ("Stage III").
- DIFFERENTIATION IS DIFFERENT -- read carefully: differentiation is a real, visually assessable
  property (how organized/mature the cells look), not just a label. If the source explicitly
  describes differentiation ("poorly differentiated", "loss of differentiation",
  "undifferentiated"), keep that information but rephrase it as a morphological observation
  rather than a grading term, e.g. "poorly differentiated" -> "the cells show limited
  differentiation" / "marked loss of differentiation and organization". Do NOT delete this
  content, and do NOT invent it when the source says nothing about differentiation.
- INVASION DEPTH IS ALSO DIFFERENT -- same logic as differentiation: depth or extent of invasion
  into a named tissue layer or structure (e.g. "invasion of the submucosa", "extension into the
  muscularis propria", "breaches the serosa", "invasion through the fibrous capsule", "extension
  into perivesical/periadrenal fat") is a real, directly observable histological finding, not
  merely a label -- keep it as a described invasion pattern, even when the source ALSO separately
  states the formal stage code derived from that same depth (e.g. "pT1"). Strip only the code
  itself (the letters/numbers), never the descriptive sentence about which layer was invaded. Do
  not treat the descriptive sentence as "just the same thing as the code" and discard both --
  they are different pieces of information; only the code is forbidden.
- DO NOT mention margins or resection borders, or specimen/sample extent
  ("margin", "surgical margin", "entire specimen", "whole specimen").
- DO NOT mention immunohistochemistry, biomarkers, or molecular/genetic test results
  (staining results, receptor status, specific antibody names, mutation/amplification status).
  This is different from a cell-type description: "neuroendocrine-appearing cells" (visual, keep)
  is not the same as "positive for synaptophysin" (a lab test result, remove).
- DO NOT mention metastasis as a staging conclusion, or explicit site-of-origin wording.

MICROSCOPIC SCOPE ONLY -- this is a separate rule from diagnostic-leakage removal:
These captions pair with a whole-slide microscopic image (a thin, stained tissue section viewed
under magnification) -- NOT a photograph of the excised specimen before sectioning. Anything that
describes the GROSS specimen -- how it looked or felt to the naked eye or hand before processing
-- cannot be observed in a microscopic image and must be removed, even when it reveals nothing
about the diagnosis. This is simply the wrong kind of information for what the image actually
shows, not a leakage issue.
- DO NOT mention any size or dimension of the specimen or mass ("measuring 5.0 cm", "3.5 x 2 x 2
  cm", "up to 13.5 cm", or any other cm/mm measurement) -- drop it entirely, do not paraphrase it
  ("a large mass measuring...") or keep a vaguer version of it.
- DO NOT mention gross consistency or texture ("soft, fleshy", "firm", "rubbery", "friable",
  "spongy", "gritty") -- these describe how the whole specimen feels, not a microscopic finding.
- DO NOT describe the gross cut surface, including its color ("tan-brown cut surface", "pale
  yellow to tan cut surface", "hemorrhagic maroon cut surface").
- Hemorrhage, necrosis, and calcification may still be kept -- but only phrase them as a
  microscopic tissue finding ("areas of hemorrhage are present within the tissue"), never as a
  description of the gross specimen's outward look or feel.

MICROSCOPIC-SCOPE WORKED EXAMPLES:
- Invalid: source "...characterized by a soft, fleshy tumor tissue with multiple foci of
  hemorrhage, measuring up to 5.0 cm..." -> WRONG strict text "The tissue displays a soft, fleshy
  appearance with multiple areas of hemorrhage." ("soft, fleshy appearance" is a gross/tactile
  specimen description and must be dropped; the size is also correctly absent here, good, but the
  texture word slipped through). Correct: "The tissue shows multiple areas of hemorrhage."
- Invalid: source "...a primary leiomyosarcoma with an estimated maximum tumor size of 13.5 cm,
  breaching the serosa and extending into the endometrium..." -> WRONG strict text "A large mass
  measuring up to 13.5 cm breaches the serosal surface and extends into the adjacent lining."
  (keeps the gross size measurement). Correct: "The lesion breaches the serosal surface and
  extends into the adjacent lining."

KEEP as much genuine visual/structural information as the source actually contains -- do not
under-report real morphology for the sake of sounding generic or "safe":
- cellular and tissue architecture (how cells/structures are arranged -- cords, nests, sheets,
  papillary, cribriform, solid, pseudoacinar, tubular, ductal-pattern, micropapillary,
  trabecular, etc.)
- cell type, lineage, or appearance ("melanocytic", "neuroendocrine-appearing", "glial-appearing",
  "spindle cell", "clear cell", etc.) -- but only when the source independently describes it as
  something seen, not when it is merely the entity name's root translated into an adjective (see
  "DO NOT TRANSLATE THE ENTITY NAME INTO ITS OWN DEFINITION" below)
- differentiation and organization, rephrased as an observation when explicitly stated (see above)
- cell-level morphology (size, shape, nuclear features, cytoplasm, mitotic activity)
- growth pattern and spatial relationships genuinely visible in tissue (infiltration into a
  named adjacent structure, invasion depth into a named tissue layer, necrosis, invasion of
  nerves or vessels) described as an observed finding, not as a staging or grading criterion --
  this counts as sufficient content on its own, it does not need to be paired with separate
  cytological/architectural detail to be worth keeping (see INSUFFICIENT-SOURCE RULE)
- any other purely structural/visual detail present in the source

DO NOT TRANSLATE THE ENTITY NAME INTO ITS OWN DEFINITION -- this is a distinct failure mode from
inventing new content, and it is just as serious: mechanically converting the entity name's root
into its etymological/defining descriptive form is NOT extraction, even though it sounds
morphological. Examples of this specific failure: "adenocarcinoma" -> "glandular structures" /
"glandular architecture"; "squamous cell carcinoma" -> "squamoid morphology"; "oligodendroglioma"
-> "oligodendroglial-like cells". Doing this by default, whenever that entity name appears,
defeats the whole purpose: "glandular" becomes just as reliable a stand-in for "adenocarcinoma"
as the word itself, so nothing was actually protected -- the leakage is merely disguised in
different words. The test: would a pathologist need to have actually looked at THIS slide to
write this detail, or could they write it purely from knowing the diagnosis name, without ever
looking at the tissue? If the source gives you nothing beyond naming the entity (plus maybe
grade/stage/size), you may NOT manufacture an architecture description by unpacking what the
entity name etymologically means. Only describe architecture/appearance that the source
independently documents as an observation.

WORKED EXAMPLES:
- Valid: source "poorly differentiated squamous carcinoma with marked pleomorphism" -> strict
  "The cells show limited differentiation and marked pleomorphism." (drops the entity name
  "carcinoma"; keeps differentiation and pleomorphism, both explicitly stated in the source --
  note "squamoid"/"squamous" is correctly NOT added, because the source adds no independent
  description of cell type beyond naming the carcinoma as squamous)
- Valid: source "adenocarcinoma with mucinous features, moderately differentiated" -> strict
  "The tissue shows moderate differentiation with prominent mucinous features." (keeps
  differentiation and "mucinous", both explicitly stated; does NOT add "glandular structures" --
  the source never described glandular architecture as an observation, only named the entity)
- Invalid: source "squamous carcinoma, grade 3." -> strict "The cells show limited
  differentiation and squamoid morphology." This is WRONG on two counts: the source never stated
  differentiation (grade 3 is a formal label, not a differentiation description) or any cell
  appearance -- both "limited differentiation" and "squamoid morphology" are pure hallucination,
  each one separately deduced from the entity name/grade rather than extracted from a real
  observation. The correct output for this source has little or nothing to hold onto (see
  INSUFFICIENT-SOURCE RULE).

IMPORTANT NEGATION RULE:
- Do not add absent or negative findings unless they are explicitly stated in the source text
  and are morphologically important.
- NEVER append a sentence like "no additional features are described" or "no specific
  microscopic features are described" after content you have already written -- if you already
  wrote genuine morphology, stop there. That kind of sentence is not useful and must never
  appear alongside real content.

INSUFFICIENT-SOURCE RULE -- read carefully, this is checked mechanically downstream:
- Some of these source excerpts are short and diagnosis-heavy with essentially no morphological
  description at all (pure diagnosis name + grade + stage, with nothing visual/structural to
  hold onto). Do not pad these with vague euphemistic language just to fill space (e.g. don't
  write "a population of cells arranged in a pattern characteristic of infiltration" when the
  source gave you nothing beyond "infiltrative carcinoma").
- Set "has_sufficient_morphology" to true if the source contains at least one genuine, concrete
  visual/structural detail that is not just a rewording of the diagnosis name -- EITHER a
  cellular/architectural detail (architecture, cell shape, nuclear features, necrosis, etc.) OR a
  described invasion/infiltration into a specific named tissue layer or anatomical structure
  (e.g. "infiltration of the submucosa", "extension into the muscularis propria", "breaches the
  serosa"). An invasion-depth finding is sufficient BY ITSELF -- it does not need to be paired
  with cytological detail, and it still counts even when the source also separately states the
  formal stage code derived from that same depth (strip only the code, keep the finding; see
  "INVASION DEPTH IS ALSO DIFFERENT" above). Apply this consistently: two sources that each give
  you nothing but one invasion-depth fact should be treated the same way as each other.
  Example: source "moderately differentiated adenocarcinoma... measuring 6 cm, with infiltration
  of the tunica submucosa. Staged as pT1 pN0 pMX" -> has_sufficient_morphology = true, strict
  text "The tissue is moderately differentiated with invasion into the submucosal layer." (do NOT
  exclude this -- the submucosal invasion is real, independent content, not merely a restatement
  of "pT1").
- DIFFERENTIATION ALONE IS NOT ENOUGH: if the ONLY extractable content is the differentiation
  level (e.g. the source is just "moderately differentiated adenocarcinoma, grade 2" with nothing
  else), that is NOT sufficient by itself -- set "has_sufficient_morphology" to false. A sentence
  that says nothing but "the tissue shows a moderate degree of differentiation" is nearly
  identical across hundreds of unrelated images (only 3-4 differentiation levels exist across the
  whole dataset) and provides no real per-image signal -- it is exactly the same duplicate,
  uninformative-caption problem as having no morphology at all. Differentiation only counts
  toward sufficiency when it accompanies at least one other genuine structural/architectural/
  cellular detail.
- If it does not, set "has_sufficient_morphology" to false and leave
  "slide_reports_no_diagnosis_strict" as an empty string. Do NOT write a fallback sentence in
  that case -- an empty string is the correct output when there is truly nothing to describe.
  This is not a failure -- it is the honest, expected outcome for some rows, and it is handled
  separately downstream.

STYLE RULES:
- Write readable, compact, medically natural prose -- not telegraphic fragments.
- Do not repeat the same feature multiple times.
- Avoid sounding like a final pathology diagnosis.
"""


def build_retry_prompt(matched_terms: list[str]) -> str:
    joined = ", ".join(sorted(set(matched_terms))[:20])
    return PROMPT_NO_DIAGNOSIS_STRICT + f"""

RETRY MODE:
Your previous output leaked forbidden diagnostic/grading/staging/organ/biomarker information.
Specifically, it still contained (or matched) one or more of: {joined}.

You must now be stricter and specifically avoid every one of those terms and anything that
implies them. If needed, produce a shorter and more conservative summary -- it is better to be
brief than to leak diagnostic information.
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "has_sufficient_morphology": {"type": "boolean"},
        "slide_reports_no_diagnosis_strict": {"type": "string"},
    },
    "required": ["has_sufficient_morphology", "slide_reports_no_diagnosis_strict"],
    "additionalProperties": False,
}


def build_payload(source_text: str) -> str:
    return f"""
TASK:
Rewrite this pathology slide report excerpt into a strict morphology-only summary with no
diagnostic leakage, following all the rules above.

SOURCE_EXCERPT:
{one_paragraph(source_text)}

IMPORTANT INSTRUCTIONS:
- Remove any explicit or implicit final diagnosis wording, tumor/entity name, and organ-of-origin.
- Remove grading, staging, margins, specimen extent, metastasis, and immunohistochemistry/molecular
  wording.
- Keep every genuine microscopic, cytologic, architectural, and invasion-pattern detail present in
  the source -- do not drop real morphology just to shorten the text.
- Never invent morphology that is not present in the source.
- The final text must remain readable, coherent, and medically natural.
"""


# ---------------- LLM CALL ----------------

def call_llm(client: OpenAI, model: str, system_prompt: str, payload: str) -> dict:
    r = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "tcga_no_diagnosis_strict_schema",
                "schema": SCHEMA,
                "strict": True,
            }
        },
        temperature=0.0,
        max_output_tokens=400,
        store=False,
    )
    return json.loads(r.output_text)


def call_llm_with_backoff(client: OpenAI, model: str, system_prompt: str, payload: str) -> dict:
    while True:
        try:
            return call_llm(client, model, system_prompt, payload)
        except RateLimitError as e:
            wait = parse_retry_seconds(str(e))
            time.sleep((wait + 2) if wait else 30)
        except (APITimeoutError, APIError):
            time.sleep(10)


# Por debajo de esto, aunque has_sufficient_morphology venga en true, se trata como
# "insuficiente" igualmente -- red de seguridad por si el modelo marca true pero en
# la practica no escribe nada aprovechable.
MIN_USEFUL_OUTPUT_CHARS = 20


def generate_validated_summary(
    client: OpenAI, model: str, row: pd.Series, max_attempts: int = 3
) -> dict:
    source_text = row[SOURCE_COLUMN]
    terms = dynamic_forbidden_terms(row)
    removed_from_source = sorted(set(has_forbidden_leakage(one_paragraph(source_text), terms)))

    base_result = {
        "slide_id": row["slide_id"],
        "case_id": row.get("case_id"),
        "project_id": row.get("project_id"),
        "original_slide_reports": row[SOURCE_COLUMN],
        "removed_terms": removed_from_source,
    }

    last_text = ""
    last_matched: list[str] = []
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        system_prompt = (
            build_retry_prompt(last_matched) if attempt > 1 else PROMPT_NO_DIAGNOSIS_STRICT
        )
        payload = build_payload(source_text)
        parsed = call_llm_with_backoff(client, model, system_prompt, payload)
        text = ensure_period(one_paragraph(parsed.get("slide_reports_no_diagnosis_strict", "")))
        has_morphology = bool(parsed.get("has_sufficient_morphology")) and len(text) >= MIN_USEFUL_OUTPUT_CHARS

        if not has_morphology:
            return {
                **base_result,
                "excluded_insufficient_source": True,
                "rewritten_text": "",
                "leakage_flag_after_retries": False,
                "remaining_matched_terms": [],
                "attempts_used": attempt,
            }

        last_text = text
        last_matched = has_forbidden_leakage(text, terms)
        if not last_matched:
            break

    return {
        **base_result,
        "excluded_insufficient_source": False,
        "rewritten_text": last_text,
        "leakage_flag_after_retries": bool(last_matched),
        "remaining_matched_terms": sorted(set(last_matched)),
        "attempts_used": attempts_used,
    }


# ---------------- CSV PLUMBING ----------------

def load_master(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=CSV_SEP, low_memory=False)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed|^$")]
    return df


def select_universe(df: pd.DataFrame, ids_csv: Path | None) -> pd.DataFrame:
    if ids_csv is None:
        return df
    ids = pd.read_csv(ids_csv)["slide_id"]
    subset = df[df["slide_id"].isin(ids)].copy()
    missing = set(ids) - set(subset["slide_id"])
    if missing:
        print(f"[aviso] {len(missing)} slide_id de {ids_csv} no aparecen en el CSV maestro.", file=sys.stderr)
    return subset


def stratified_pilot(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    n = min(n, len(df))
    frac_per_group = n / len(df)
    sample = (
        df.groupby("project_id", group_keys=False)
        .apply(lambda g: g.sample(max(1, round(len(g) * frac_per_group)), random_state=seed),
               include_groups=False)
    )
    sample["project_id"] = df.loc[sample.index, "project_id"]
    if len(sample) > n:
        sample = sample.sample(n, random_state=seed)
    return sample.reset_index(drop=True)


# ---------------- MAIN ----------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, required=DEFAULT_CSV is None,
                   help="CSV maestro (separador ';'), ver docs/DATA_FORMAT.md.")
    p.add_argument("--strict-common-ids", type=Path, default=DEFAULT_STRICT_COMMON_IDS,
                   help="Universo opcional: slide_id de la cohorte a procesar (omitir = --all).")
    p.add_argument("--all", action="store_true",
                   help="Usar los 10.108 slides del CSV maestro en vez del subset strict-common.")
    p.add_argument("--pilot", type=int, default=0,
                   help="Muestrear N filas (estratificadas por project_id) para revisar calidad "
                        "antes de lanzar la tanda completa.")
    p.add_argument("--pilot-review-out", type=Path, default=None,
                   help="Con --pilot: en vez de fusionar en el CSV maestro, vuelca a este CSV de "
                        "revision (por defecto /tmp/tcga_no_diagnosis_pilot_review.csv) y no toca --csv.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--out", type=Path, default=None,
                   help="Por defecto sobrescribe --csv (con escritura atomica). Usa esto para "
                        "escribir en otra ruta y no tocar el original.")
    p.add_argument("--audit-json", type=Path, default=None,
                   help="Por defecto, junto al csv de salida con sufijo _audit.json. Un objeto por "
                        "fila procesada con lo que se quito/cambio y si sigue habiendo posible fuga.")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerar tambien filas que ya tienen texto en la columna nueva "
                        "(por defecto se saltan, para poder reanudar una tanda cortada).")
    p.add_argument("--checkpoint-every", type=int, default=25)
    args = p.parse_args()

    df = load_master(args.csv)
    ids_csv = None if args.all else args.strict_common_ids
    universe = select_universe(df, ids_csv)

    is_pilot = bool(args.pilot)
    if is_pilot:
        universe = stratified_pilot(universe, args.pilot, seed=args.seed)
        print(f"Piloto: {len(universe)} filas estratificadas por project_id.")
    else:
        print(f"Tanda completa: {len(universe)} filas candidatas (antes de aplicar reanudacion).")

    universe = universe[universe[SOURCE_COLUMN].notna() & (universe[SOURCE_COLUMN].str.strip() != "")]

    if NEW_COLUMN not in df.columns:
        df[NEW_COLUMN] = pd.NA

    out_path = args.out or args.csv
    audit_path = args.audit_json or out_path.with_name(out_path.stem + "_audit.json")
    review_csv_path = args.pilot_review_out or Path("/tmp/tcga_no_diagnosis_pilot_review.csv")

    # se carga la auditoria previa (si existe) para que la reanudacion tambien
    # sepa que filas ya se marcaron como "sin morfologia suficiente" -- esas no
    # dejan texto en la columna del CSV, asi que sin esto se reprocesarian en
    # cada relanzamiento del script.
    audit_rows: list[dict] = json.loads(audit_path.read_text()) if audit_path.exists() else []
    already_audited_ids = {r["slide_id"] for r in audit_rows}
    audit_by_id = {r["slide_id"]: r for r in audit_rows}

    if not args.overwrite:
        # las filas ya auditadas en una corrida anterior (p.ej. un --pilot, que
        # nunca escribe al CSV maestro) pueden tener el texto ya generado en el
        # JSON pero todavia NO en la columna del CSV -- se rellena aqui antes de
        # saltarlas, para no perder (ni volver a pagar) trabajo ya hecho.
        n_backfilled = 0
        for sid in already_audited_ids & set(universe["slide_id"]):
            r = audit_by_id[sid]
            if r["excluded_insufficient_source"]:
                continue
            mask = df["slide_id"] == sid
            if mask.any() and not has_useful_text(df.loc[mask, NEW_COLUMN].iloc[0]):
                df.loc[mask, NEW_COLUMN] = r["rewritten_text"]
                n_backfilled += 1
        if n_backfilled:
            print(f"Recuperadas {n_backfilled} filas ya generadas en una corrida previa (auditoria) "
                  f"-> columna del CSV rellenada sin volver a llamar a la API.")

        already_done_mask = df["slide_id"].isin(universe["slide_id"]) & df[NEW_COLUMN].apply(has_useful_text)
        skip_ids = set(df.loc[already_done_mask, "slide_id"]) | (already_audited_ids & set(universe["slide_id"]))
        universe = universe[~universe["slide_id"].isin(skip_ids)]
        if skip_ids:
            print(f"Reanudacion: {len(skip_ids)} filas ya procesadas (con texto o marcadas sin "
                  f"morfologia suficiente) -> se saltan (usa --overwrite para regenerarlas).")
    else:
        audit_rows = [r for r in audit_rows if r["slide_id"] not in set(universe["slide_id"])]

    if universe.empty:
        print("Nada que procesar (todo ya generado, o el universo/filtro no selecciona filas).")
        return

    client = OpenAI()
    n_leaked = 0
    n_excluded = 0

    for i, row in enumerate(tqdm(list(universe.itertuples(index=False, name="Row")),
                                  desc="tcga_no_diagnosis_strict"), start=1):
        row_series = pd.Series(row._asdict())
        result = generate_validated_summary(client, args.model, row_series, max_attempts=args.max_attempts)
        if result["excluded_insufficient_source"]:
            n_excluded += 1
        elif result["leakage_flag_after_retries"]:
            n_leaked += 1

        if not result["excluded_insufficient_source"]:
            df.loc[df["slide_id"] == result["slide_id"], NEW_COLUMN] = result["rewritten_text"]
        audit_rows.append(result)

        if args.checkpoint_every and i % args.checkpoint_every == 0:
            safe_write_json(audit_rows, audit_path)
            if not is_pilot:
                safe_write_csv(df, out_path)

    safe_write_json(audit_rows, audit_path)

    if is_pilot:
        review_df = pd.DataFrame([{
            "slide_id": r["slide_id"],
            "project_id": r["project_id"],
            SOURCE_COLUMN: r["original_slide_reports"],
            NEW_COLUMN: r["rewritten_text"],
            "excluded_insufficient_source": r["excluded_insufficient_source"],
            "leakage_flag_after_retries": r["leakage_flag_after_retries"],
            "attempts_used": r["attempts_used"],
        } for r in audit_rows if r["slide_id"] in set(universe["slide_id"])])
        review_df.to_csv(review_csv_path, index=False)
        print(f"\nPiloto volcado para revision manual en {review_csv_path} ({len(review_df)} filas) "
              f"y {audit_path} (detalle completo: texto original, terminos quitados, etc). "
              f"No se ha tocado {args.csv}.")
    else:
        safe_write_csv(df, out_path)
        print(f"\nColumna '{NEW_COLUMN}' actualizada para {len(universe) - n_excluded} filas -> {out_path}")

    print(f"Auditoria completa en {audit_path}.")
    print(f"  {n_excluded}/{len(universe)} filas excluidas por no tener morfologia real que describir "
          f"(quedan sin valor en '{NEW_COLUMN}', no se ha inventado texto de relleno).")
    print(f"  {n_leaked}/{len(universe)} filas siguen con posible fuga tras {args.max_attempts} "
          f"intentos -- revisalas a mano.")


if __name__ == "__main__":
    main()
