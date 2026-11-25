"""Experiment configuration: every path comes from a YAML file (see configs/paths.example.yaml)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

BACKBONES = ["KEEP", "CONCH", "MUSK", "PATHO_CLIP"]
TEXT_CONDITIONS = ["original", "morphology_only", "diagnosis_only"]
TEXT_LABEL = {"original": "Original", "morphology_only": "Morphology-only", "diagnosis_only": "Diagnosis-only"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def output_dir(cfg: dict[str, Any], *parts: str) -> Path:
    """<output_dir>/<parts...>, created on demand."""
    d = Path(cfg["output_dir"]).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_argument_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="YAML file with data paths (see configs/paths.example.yaml)")
    return p
