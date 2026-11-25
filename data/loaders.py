"""
Loading of embeddings, labels and partitions from the paths declared in the YAML config.

Expected on-disk formats (no data is shipped with the repository):

* patch embeddings: ``<patch_dir>/<slide_id>.npy``, (N_i, d) array per WSI. For AI4SKIN the
  directory can also be given inside a zip archive as ``zip:<archive.zip>!<inner/dir>``.
* report embeddings: ``<text_dir>/<slide_id>.npy``, (d,) array per report and text condition.
* pooled cache: ``.npz`` with ``case_ids``, ``mean_pool``, ``max_pool`` (``data/build_pool_cache.py``).
* prototype banks: ``<prototype_dir>/<slide_id>_prototype.npz`` (TCGA) or ``<case_id>.npz``
  (AI4SKIN), key ``prototypes`` (``data/build_prototype_banks.py``).

Two L2 normalisations exist in the original experiments and are kept to reproduce every
number bit for bit: ``"vector"`` (each 1-D vector divided by max(||v||, 1e-12)) and
``"rows"`` (stacked matrix divided by its clipped row norms). They differ at most by one
floating-point ulp.
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpgr.similarity import l2_normalize_rows

EPS = 1e-12


# ----------------------------------------------------------------------------- generic readers
class PatchStore:
    """Reads ``<id>.npy`` patch matrices from a directory or from ``zip:<archive>!<inner_dir>``."""

    def __init__(self, spec: str):
        self.spec = str(spec)
        if self.spec.startswith("zip:"):
            archive, inner = self.spec[4:].split("!", 1)
            self._zip = zipfile.ZipFile(archive)
            self._inner = inner.rstrip("/")
        else:
            self._zip = None
            self._dir = Path(self.spec)

    def load(self, item_id: str) -> np.ndarray:
        if self._zip is not None:
            return np.load(io.BytesIO(self._zip.read(f"{self._inner}/{item_id}.npy")))
        return np.load(self._dir / f"{item_id}.npy")

    def ids(self) -> list[str]:
        if self._zip is not None:
            pre = self._inner + "/"
            return sorted(n[len(pre):-4] for n in self._zip.namelist() if n.startswith(pre) and n.endswith(".npy"))
        return sorted(p.stem for p in self._dir.glob("*.npy"))


def load_text_vectors(text_dir: str | Path, ids: list[str], mode: str = "vector") -> np.ndarray:
    """(n, d) float64 report embeddings in ``ids`` order, L2-normalised ('vector' or 'rows')."""
    raw = [np.load(Path(text_dir) / f"{c}.npy").astype(np.float64).reshape(-1) for c in ids]
    if mode == "vector":
        return np.stack([v / max(np.linalg.norm(v), EPS) for v in raw])
    if mode == "rows":
        return l2_normalize_rows(np.stack(raw))
    raise ValueError(mode)


def load_prototype_banks(paths: list[Path], renormalize: bool = False) -> list[np.ndarray]:
    """Prototype banks as float64; optionally re-normalise their rows (AI4SKIN stress test)."""
    banks = [np.load(p, allow_pickle=True)["prototypes"].astype(np.float64) for p in paths]
    return [l2_normalize_rows(b) for b in banks] if renormalize else banks


# ----------------------------------------------------------------------------- TCGA
class TCGA:
    """Accessors for the TCGA cohort declared under ``tcga:`` in the config."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg["tcga"]

    @lru_cache(maxsize=1)
    def split(self) -> pd.DataFrame:
        return pd.read_csv(self.cfg["split_csv"])

    def ids(self, partition: str | list[str]) -> list[str]:
        parts = [partition] if isinstance(partition, str) else partition
        s = self.split()
        return s[s.split.isin(parts)]["slide_id"].tolist()

    def complete_cohort_ids(self) -> list[str]:
        return pd.read_csv(self.cfg["complete_cohort_csv"])["slide_id"].tolist()

    @lru_cache(maxsize=1)
    def labels(self) -> pd.DataFrame:
        return pd.read_csv(self.cfg["labels_csv"]).set_index("slide_id")

    def label_map(self, ids: list[str], level: str = "subtype") -> dict[str, str]:
        col = {"subtype": "OncotreeCode", "organ": "project_id"}[level]
        return self.labels()[col].reindex(ids).to_dict()

    def case_ids(self, ids: list[str]) -> np.ndarray:
        return self.labels()["case_id"].reindex(ids).to_numpy()

    def backbone(self, name: str) -> dict[str, Any]:
        return self.cfg["backbones"][name]

    def pooled(self, backbone: str, ids: list[str], kind: str = "mean_pool") -> np.ndarray:
        """Row-normalised float64 mean- or max-pooled WSI vectors in ``ids`` order."""
        cache = np.load(self.backbone(backbone)["pool_cache"], allow_pickle=True)
        ix = {c: i for i, c in enumerate(cache["case_ids"])}
        return l2_normalize_rows(cache[kind].astype(np.float64))[[ix[c] for c in ids]]

    def text(self, backbone: str, condition: str, ids: list[str], mode: str = "vector") -> np.ndarray:
        return load_text_vectors(self.backbone(backbone)["text_dirs"][condition], ids, mode)

    def prototype_banks(self, backbone: str, ids: list[str]) -> list[np.ndarray]:
        d = Path(self.backbone(backbone)["prototype_dir"])
        return load_prototype_banks([d / f"{c}_prototype.npz" for c in ids])

    def patches(self, backbone: str, ids: list[str]) -> list[np.ndarray]:
        """Row-normalised float64 patch matrices."""
        store = PatchStore(self.backbone(backbone)["patch_dir"])
        return [l2_normalize_rows(store.load(c).astype(np.float64)) for c in ids]

    def trained(self, key: str) -> str:
        return self.cfg["trained_outputs"][key]


# ----------------------------------------------------------------------------- AI4SKIN
class AI4SKIN:
    """Accessors for the AI4SKIN cohort declared under ``ai4skin:`` in the config."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg["ai4skin"]

    @lru_cache(maxsize=1)
    def folds(self) -> pd.DataFrame:
        return pd.read_csv(self.cfg["folds_csv"])

    def cohort_ids(self) -> list[str]:
        """The 276 cases in cohort-manifest order."""
        return pd.read_csv(self.cfg["cohort_csv"])["case_id"].tolist()

    def cohort_label_map(self) -> dict[str, str]:
        rows = pd.read_csv(self.cfg["cohort_csv"])
        mapping = pd.read_csv(self.cfg["class_mapping_csv"])
        m = dict(zip(mapping.class_name_corregida, mapping.class_name_en))
        return {c: m[k] for c, k in zip(rows.case_id, rows.class_name_corregida)}

    def fold_ids(self, fold: int, role: str) -> list[str]:
        f = self.folds()
        return list(f[(f.outer_fold == fold) & (f.role == role)].case_id)

    def fold_label_map(self, fold: int, role: str) -> dict[str, str]:
        f = self.folds()
        sub = f[(f.outer_fold == fold) & (f.role == role)]
        return dict(zip(sub.case_id, sub.class_name_en))

    def backbone(self, name: str) -> dict[str, Any]:
        return self.cfg["backbones"][name]

    def patch_store(self, backbone: str) -> PatchStore:
        return PatchStore(self.backbone(backbone)["patch_dir"])

    def text(self, backbone: str, condition: str, ids: list[str], mode: str = "vector") -> np.ndarray:
        return load_text_vectors(self.backbone(backbone)["text_dirs"][condition], ids, mode)

    def prototype_banks(self, backbone: str, ids: list[str], renormalize: bool = False) -> list[np.ndarray]:
        d = Path(self.backbone(backbone)["prototype_dir"])
        return load_prototype_banks([d / f"{c}.npz" for c in ids], renormalize)

    def trained(self, method: str, fold: int) -> Path:
        """Exported test embeddings of a trained / slide-level model for one fold
        (``ai4skin.trained_outputs.<method>`` is a path template with ``{fold}``)."""
        return Path(self.cfg["trained_outputs"][method].format(fold=fold))
