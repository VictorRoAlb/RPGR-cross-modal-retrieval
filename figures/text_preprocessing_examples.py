#!/usr/bin/env python3
"""
text_preprocessing_examples.py
=================================
Reproduces Figure A.1: the three textual conditions used throughout the
benchmark (Original / Strict / Diagnosis-only, see Table 1) shown for four
representative TCGA cases. Diagnosis-only is literally `cancer_type_detailed`
(the short diagnostic label), the opposite extreme from Strict -- not a
keyword list.

The four cases and their text are hardcoded, not regenerated -- they are
the exact examples already verified in the paper. Requires weasyprint.

Usage:
  python3 text_preprocessing_examples.py --output-dir ../outputs
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from weasyprint import HTML

CASES = [
    {
        "label": "BLCA · bladder",
        "slide_id": "TCGA-ZF-AA4N-01Z-00-DX1.F176510E-CF8B-4681-9CDE-B6210D8DC88F",
        "orig": 'The slide from the bladder shows a poorly differentiated carcinoma (G3, pT2) with a solid growth pattern invading the muscularis propria. Features include laminated layers of keratin indicating squamous differentiation and large areas of necrosis. No transitional urothelium is identified, and non-dysplastic squamous epithelium is present. No vascular invasion is identified.',
        "strict": 'The tissue displays a solid growth pattern with invasion into the muscularis propria. The cells show limited differentiation, and there are laminated layers of keratin indicating squamous differentiation, along with large areas of necrosis. No transitional-type epithelium is identified, while non-dysplastic squamous epithelium is present. No vascular invasion is observed.',
        "dxonly": "Bladder Urothelial Carcinoma",
        "dx_spans": ["carcinoma (G3, pT2)"],
    },
    {
        "label": "GBM · brain",
        "slide_id": "TCGA-06-0138-01Z-00-DX1.212817bd-0fa0-4503-ae5b-4eaee8ee95e5",
        "orig": 'The slide from the brain shows a glioblastoma characterized by diffuse infiltration and extensive effacement of the cerebrum with nuclear anaplasia, mitotic figures, microvascular proliferation, and necrosis. The neoplasm exhibits an astrocytic phenotype, predominantly gemistocytic, with focal features suggestive of an oligodendroglial component. The MIB-1 proliferation index is approximately 20%.',
        "strict": 'The tissue demonstrates diffuse infiltration and extensive effacement with marked nuclear anaplasia, frequent mitotic figures, prominent microvascular proliferation, and areas of necrosis. The cellular population displays an astrocytic appearance, predominantly gemistocytic, with focal regions exhibiting features reminiscent of an oligodendroglial component.',
        "dxonly": "Glioblastoma Multiforme",
        "dx_spans": ["glioblastoma", "The MIB-1 proliferation index is approximately 20%."],
    },
    {
        "label": "IDC · breast",
        "slide_id": "TCGA-AO-A0J5-01Z-00-DX1.20C14D0C-1A74-4FE9-A5E6-BDDCB8DE7714",
        "orig": 'The slide from the right breast shows invasive carcinoma, histologic grade II/III to grade III/III with micropapillary features, and focal extracellular mucin. Ductal carcinoma in situ (DCIS) is also identified, characterized by solid cribriform, micropapillary, and flat types with high nuclear grade and extensive necrosis. There is lobular involvement by DCIS, which constitutes less than or equal to 25% of the total tumor mass. Vascular invasion is present, and invasive carcinoma involves the skin and attached skeletal muscle. Metastatic carcinoma is noted in 2 out of 2 level I lymph nodes. The tumor measures at least 10 cm in largest dimension.',
        "strict": 'The tissue displays invasive growth with micropapillary features and focal areas of extracellular mucin. Noninvasive epithelial proliferations are present, exhibiting solid, cribriform, micropapillary, and flat architectural patterns, accompanied by high nuclear grade and extensive necrosis. There is involvement of lobular structures by these noninvasive components, comprising up to a quarter of the total lesion. Vascular invasion is observed, and the invasive process extends into the skin and adjacent skeletal muscle.',
        "dxonly": "Breast Invasive Ductal Carcinoma",
        "dx_spans": [
            "carcinoma, histologic grade II/III to grade III/III",
            "Ductal carcinoma in situ (DCIS)",
            "carcinoma involves",
            "Metastatic carcinoma is noted in 2 out of 2 level I lymph nodes. The tumor measures at least 10 cm in largest dimension.",
        ],
    },
    {
        "label": "SSRCC · stomach",
        "slide_id": "TCGA-VQ-A8DZ-01Z-00-DX1.3C4CB10D-8C8B-4828-B7D4-7058BDF4D39F",
        "orig": 'The slide from the gastric antrum shows a poorly differentiated adenocarcinoma with mucosecretor mucocellular characteristics, featuring "signet ring" cells. The tumor is classified as diffuse pattern according to Lauren’s classification, with extensive involvement of the adjacent adipose tissue, neural infiltration, and lymphatic vascular invasion.',
        "strict": 'The tissue displays poorly differentiated cells with mucosecretory and mucocellular features, including cells characterized by abundant intracytoplasmic mucin that displaces the nucleus peripherally. There is extensive infiltration into adjacent adipose tissue, as well as evidence of neural and lymphatic vascular invasion.',
        "dxonly": "Signet Ring Cell Carcinoma of the Stomach",
        "dx_spans": [
            "adenocarcinoma",
            '"signet ring"',
            "The tumor is classified as diffuse pattern according to Lauren’s classification,",
        ],
    },
]


def highlight_diagnostic(text: str, spans: list[str]) -> str:
    """Wraps each literal substring in `spans` (diagnostic nomenclature:
    disease name, grade/stage, named classification -- not stylistic
    rewording) in a <span class=dxword>."""
    out = html.escape(text)
    for span in spans:
        esc = html.escape(span)
        out = out.replace(esc, f'<span class="dxword">{esc}</span>')
    return out


def build_html() -> str:
    rows_html = []
    for case in CASES:
        rows_html.append(f"""
        <tr>
          <td class="tag"><div class="tagbox">{html.escape(case['label'])}<div class="slideid">{html.escape(case['slide_id'])}</div></div></td>
          <td class="cell text"><div class="boxlabel">Original</div>{highlight_diagnostic(case['orig'], case['dx_spans'])}</td>
          <td class="arrowcell">&#10230;</td>
          <td class="cell text"><div class="boxlabel">Strict</div>{html.escape(case['strict'])}</td>
          <td class="arrowcell">&#10230;</td>
          <td class="cell dxonly"><div class="boxlabel">Diagnosis-only</div><div class="dxtext">{html.escape(case['dxonly'])}</div></td>
        </tr>
        """)

    return f"""
<!doctype html><html><head><meta charset="utf-8">
<style>
@page {{ size: 1500px 970px; margin: 0; }}
body {{ margin:0; padding:28px 30px; font-family: 'DejaVu Sans', Arial, sans-serif;
        color:#1a1a1a; background:#ffffff; }}
h1 {{ font-size:18px; font-weight:700; margin:0 0 4px; }}
.sub {{ font-size:11.5px; color:#555; margin:0 0 16px; max-width:1420px; }}
table {{ border-collapse: separate; border-spacing: 0 14px; width:100%; table-layout:fixed; }}
col.tagcol {{ width: 168px; }}
col.textcol {{ width: 430px; }}
col.arrowcol {{ width: 40px; }}
col.dxcol {{ width: 220px; }}
td {{ vertical-align: top; padding: 0 8px; }}
.tag {{ vertical-align: middle; }}
.tagbox {{ font-size:12px; font-weight:700; color:#333; background:#eef0f3;
           border-radius:8px; padding:8px 6px; text-align:center; line-height:1.3; }}
.slideid {{ margin-top:6px; font-family:'DejaVu Sans Mono', monospace; font-size:7.6px;
            font-weight:400; color:#777; word-break:break-all; line-height:1.35; }}
.cell.text {{ border:1.6px solid #333; border-radius:12px; padding:10px 12px 12px;
              font-size:12px; line-height:1.55; background:#fff; }}
.boxlabel {{ font-size:9.5px; font-weight:700; text-transform:uppercase;
             letter-spacing:.04em; color:#888; margin-bottom:5px; }}
.arrowcell {{ vertical-align:middle; text-align:center; font-size:24px; color:#444; }}
.cell.dxonly {{ vertical-align:middle; border:1.8px solid #9b2247; border-radius:12px;
                padding:14px 12px; background:#fdf1f5; text-align:center; }}
.dxtext {{ font-size:13.5px; font-weight:700; color:#9b2247; line-height:1.35; }}
.dxword {{ color:#c81e3a; font-weight:700; }}
</style></head>
<body>
<h1>The three textual conditions across representative TCGA cases</h1>
<p class="sub">Four representative TCGA cases showing the three textual settings used
throughout the benchmark (Section 4.2). <b>Original</b> retains the complete
report-derived text. <b>Strict</b> removes explicit diagnostic cues while preserving
morphological description. <b>Diagnosis-only</b> retains exclusively the short
diagnostic label (<code>cancer_type_detailed</code>) and removes all morphological
content — the opposite extreme of Strict. In the Original column, <span
style="color:#c81e3a;font-weight:700">red</span> marks explicit diagnostic
nomenclature (disease names, grading/staging, named classification systems)
rather than histological description — exactly the content that Strict
removes.</p>
<table>
<colgroup><col class="tagcol"><col class="textcol"><col class="arrowcol">
<col class="textcol"><col class="arrowcol"><col class="dxcol"></colgroup>
{"".join(rows_html)}
</table>
</body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    html_path = args.output_dir / "text_preprocessing_examples.html"
    pdf_path = args.output_dir / "text_preprocessing_examples.pdf"
    html_path.write_text(build_html())
    HTML(str(html_path)).write_pdf(str(pdf_path))
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
