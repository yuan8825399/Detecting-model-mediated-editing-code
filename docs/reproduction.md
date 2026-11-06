# Reproducing the analyses

## Setup

```bash
git clone <repository-url>
cd provenance-audit
python -m pip install -r requirements.txt
python -m pip install -e .
```

The correction run's exact analysis versions are in `requirements-reproduction.txt`.
See [corrections_20261106.md](corrections_20261106.md) for changed definitions and
the pinned BERT revision. The generation stage cannot be reconstructed from these files.

Optional, only for cached embeddings or the DOCX builder:

```bash
python -m pip install -r requirements-optional.txt
```

## Quick checks (about one minute)

```bash
pytest -q                       # data integrity, conformal machinery, metrics
python analysis/40_isolation_audit.py
```

The audit prints 51 protocols and zero document overlap. It is also run in CI.

## Rebuilding the paper's outputs

Every table and figure is committed under `results/` and `figures/`, so figures can be
rebuilt without refitting models (about 30 s):

```bash
python scripts/run_pipeline.py --only typology,figures,figure-palcp,figure-coverage
```

`typology` regenerates `processed/pairs.pkl`; the figure scripts read everything else from
the committed `results/`.

To recompute the numbers from the corpus:

```bash
python scripts/run_pipeline.py --stage core      # ~10 min, no model fitting beyond LR
```

To reproduce everything, including the detector sweeps:

```bash
python scripts/run_pipeline.py --stage all       # ~60 min on a 4-core CPU
```

`python scripts/run_pipeline.py --list` prints the stages, their estimated runtimes and
which ones are marked as heavy.

## Stage dependencies

```
04 typology ──> 11 descriptives ──> 12 figures
     │                │
     └──> 14 recalibration ──> 12 figures
05 embeddings ──> 06 main ──> 13 tables
07 transfer ──> 12 figures
18 method-probe ──> 21 palcp ──> 22 figure 8
41 verify ──> 44 figure 9
40 audit, 42 operating points: independent
```

Stages that need a fitted model read the corpus directly; none of them mutate
`data/`. Intermediates are written to `processed/`, which is git-ignored.

## What is not reproducible from this repository

The generation stage. The corpus records which system produced each variant and the
content hashes, but not the prompt text, decoding parameters, seeds, timestamps or retry
logs; see Table 1 of the paper. The corrected analysis pipeline uses scripts in
`analysis/` and the declared `exploratory/18_method_probe.py` producer, starting from
`data/benchmark_records.jsonl`. It does not reconstruct missing generation requests
or verify the original manual annotation process.

## Troubleshooting

* **`processed/emb_mean.npy` missing** — only `analysis/05_embeddings.py` and the `bert`
  row of the detector table need it. Run `--only embeddings` after installing the optional
  dependencies, or use the committed `results/main_results_bert.csv`.
* **A stage reports `SKIPPED: missing [...]`** — the runner checks its declared inputs.
  Run the producing stage first; the message names the file it was looking for.
* **Figures fail after deleting `results/`** — regenerate in pipeline order; most figure
  scripts read several `results/` tables as well as `processed/pairs.pkl`.
* **Memory** — the full pipeline peaks around 2 GB (TF-IDF matrices over 8,381 records).
