"""
Validation data of the cross-method benchmark (KEEP, Morphology-only,
subtype level): the TCGA validation partition (n=752) and the validation set of each of
the five AI4SKIN folds (55-56 cases), with the training data of the same partition / fold
as query bank for QB-Norm and DBNorm.
"""
from __future__ import annotations

import numpy as np

from data.loaders import AI4SKIN, TCGA
from rpgr.similarity import global_similarity, l2_normalize_rows, mean_pool

BACKBONE = "KEEP"
CONDITION = "morphology_only"


def _assemble(ids, label_map, mp_val, text_val, mp_train, text_train, patches_val):
    S_G = global_similarity(mp_val, text_val)
    return {
        "ids": ids, "label_map": label_map,
        "patches_val": patches_val, "text_val": text_val,
        "I2T": {"S_G": S_G,
                "S_bank_G": global_similarity(mp_train, text_val),
                "S_uni_bank_G": global_similarity(text_val, text_train)},
        "T2I": {"S_G": S_G.T,
                "S_bank_G": global_similarity(mp_val, text_train).T,
                "S_uni_bank_G": global_similarity(mp_val, mp_train)},
    }


def tcga_validation_cell(cfg) -> dict:
    tcga = TCGA(cfg)
    val_ids, train_ids = tcga.ids("val"), tcga.ids("train")
    return _assemble(val_ids, tcga.label_map(val_ids, "subtype"),
                     tcga.pooled(BACKBONE, val_ids), tcga.text(BACKBONE, CONDITION, val_ids),
                     tcga.pooled(BACKBONE, train_ids), tcga.text(BACKBONE, CONDITION, train_ids),
                     tcga.patches(BACKBONE, val_ids))


def ai4skin_validation_cell(cfg, fold: int) -> dict:
    ai = AI4SKIN(cfg)
    store = ai.patch_store(BACKBONE)
    val_ids, train_ids = ai.fold_ids(fold, "val"), ai.fold_ids(fold, "train")
    patches_val = [l2_normalize_rows(store.load(c).astype(np.float64)) for c in val_ids]
    patches_train = [l2_normalize_rows(store.load(c).astype(np.float64)) for c in train_ids]
    return _assemble(val_ids, ai.fold_label_map(fold, "val"),
                     np.stack([mean_pool(p) for p in patches_val]), ai.text(BACKBONE, CONDITION, val_ids),
                     np.stack([mean_pool(p) for p in patches_train]), ai.text(BACKBONE, CONDITION, train_ids),
                     patches_val)
