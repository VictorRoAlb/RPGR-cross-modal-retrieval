"""
Report embeddings with the frozen text encoder of each VL-PFM.

Reads a ';'-separated CSV with an id column and a text column and writes ``<out_dir>/<id>.npy``
(float32, shape (1, d)). The four encoders need different library versions; install the
matching file of ``requirements/`` in a separate environment for each one:

  KEEP        Astaxanthin/KEEP @ 28a25d95cc6ba27a7e6fab3f144e13dbafd8b21e (transformers, max_length=256)
  CONCH       MahmoodLab/conch @ f9ca9f877171a28ade80228fb195ac5d79003357 (conch_ViT-B-16)
  MUSK        xiangjx/musk @ de1ffed28608c197d2903f6fa42b491a3fbf0fb8 (musk_large_patch16_384, max_len=230)
  PATHO_CLIP  WenchuanZhang/Patho-CLIP-L @ 8c9d18432daa80f211dbcfcfcb40af42b9d1b9b7 (open_clip ViT-L-14, 77 tokens)

KEEP and MUSK return L2-normalised embeddings; PATHO-CLIP is normalised explicitly; CONCH is
stored as returned (all experiments re-normalise every report embedding).

Run with: python -m text_conditions.encode_reports --backbone KEEP --csv texts.csv \
              --id-column slide_id --text-field slide_reports_no_diagnosis_strict --out-dir <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REVISIONS = {"KEEP": "28a25d95cc6ba27a7e6fab3f144e13dbafd8b21e", "CONCH": "f9ca9f877171a28ade80228fb195ac5d79003357",
             "MUSK": "de1ffed28608c197d2903f6fa42b491a3fbf0fb8", "PATHO_CLIP": "8c9d18432daa80f211dbcfcfcb40af42b9d1b9b7"}


def load_encoder(backbone: str):
    """Returns (encode(list[str]) -> (n, d) float32 array, batch size)."""
    import torch
    rev = REVISIONS[backbone]
    if backbone == "KEEP":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True, revision=rev).eval()
        tok = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True, revision=rev)

        def encode(texts):
            with torch.no_grad():
                t = tok(texts, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
                return model.encode_text(t).detach().cpu().float().numpy().astype(np.float32)
        return encode, 8
    if backbone == "CONCH":
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        from conch.open_clip_custom.custom_tokenizer import tokenize
        model, _ = create_model_from_pretrained("conch_ViT-B-16", f"hf_hub:MahmoodLab/conch@{rev}")
        model.eval()
        tok = get_tokenizer()

        def encode(texts):
            with torch.no_grad():
                o = model.encode_text(tokenize(tok, texts))
            o = o[0] if isinstance(o, tuple) else o
            return o.detach().cpu().float().numpy().astype(np.float32)
        return encode, 4
    if backbone == "MUSK":
        from huggingface_hub import hf_hub_download
        from musk import modeling, utils  # noqa: F401  (registers the timm model)
        from timm.models import create_model
        from transformers import XLMRobertaTokenizer
        weights = hf_hub_download("xiangjx/musk", "model.safetensors", revision=rev)
        spm = hf_hub_download("xiangjx/musk", "tokenizer.spm", revision=rev)
        model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate(weights, model, "model|module", "")
        model.to(dtype=torch.float32).eval()
        tok = XLMRobertaTokenizer(spm)

        def encode(texts):
            rows = []
            with torch.no_grad():
                for t in texts:
                    ids, pad = utils.xlm_tokenizer(t, tok, max_len=230)
                    o = model(text_description=torch.tensor(ids, dtype=torch.long).unsqueeze(0),
                              padding_mask=torch.tensor(pad, dtype=torch.bool).unsqueeze(0),
                              with_head=True, out_norm=True, ms_aug=False, return_global=True)[1]
                    o = o[0] if isinstance(o, (tuple, list)) else o
                    rows.append(o.detach().cpu().float().numpy().reshape(-1))
            return np.stack(rows).astype(np.float32)
        return encode, 1
    if backbone == "PATHO_CLIP":
        import open_clip
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download("WenchuanZhang/Patho-CLIP-L", "Patho-CLIP-L.pt", revision=rev)
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=weights)
        tok = open_clip.get_tokenizer("ViT-L-14")
        model.eval()

        def encode(texts):
            with torch.no_grad():
                e = model.encode_text(tok(texts))
                e = e / e.norm(dim=-1, keepdim=True)
            return e.detach().cpu().numpy().astype(np.float32)
        return encode, 8
    raise ValueError(backbone)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=list(REVISIONS))
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--id-column", default="slide_id")
    p.add_argument("--text-field", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv, sep=";", low_memory=False)
    df = df[df[args.text_field].fillna("").str.strip() != ""]
    encode, bs = load_encoder(args.backbone)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids, texts = df[args.id_column].tolist(), df[args.text_field].astype(str).str.strip().tolist()
    for i in range(0, len(texts), bs):
        E = encode(texts[i:i + bs])
        for cid, v in zip(ids[i:i + bs], E):
            np.save(args.out_dir / f"{cid}.npy", v[None, :])
    print(f"wrote {len(ids)} embeddings to {args.out_dir}")


if __name__ == "__main__":
    main()
