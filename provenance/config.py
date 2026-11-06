"""Central path configuration.

Every path is resolved relative to the repository root, so the pipeline runs from any
working directory and after the repository is cloned anywhere. Set the environment
variables PROVENANCE_DATA_DIR / PROVENANCE_RESULTS_DIR to relocate data or outputs.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("PROVENANCE_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("PROVENANCE_RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("PROVENANCE_FIGURES_DIR", REPO_ROOT / "figures"))

#: intermediate artefacts (pickles, embeddings); never committed
PROCESSED_DIR = Path(os.environ.get("PROVENANCE_PROCESSED_DIR", REPO_ROOT / "processed"))

DATA_FILE = DATA_DIR / "benchmark_records.jsonl"
METADATA_ONLY_FILE = DATA_DIR / "benchmark_records_metadata_only.jsonl"

for _d in (RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

__all__ = [
    "REPO_ROOT", "DATA_DIR", "RESULTS_DIR", "FIGURES_DIR", "PROCESSED_DIR",
    "DATA_FILE", "METADATA_ONLY_FILE",
]
