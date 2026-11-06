#!/usr/bin/env python
"""Run the analysis pipeline.

Stages are executed in dependency order. Results for every stage are committed under
`results/`, so a fresh clone can rebuild any figure without re-running the models.

Examples
--------
    python scripts/run_pipeline.py --list
    python scripts/run_pipeline.py --stage core          # ~10 min, no model fitting
    python scripts/run_pipeline.py --only figures        # rebuild Figures 1-7
    python scripts/run_pipeline.py --stage all --skip-heavy
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from provenance.config import RESULTS_DIR, PROCESSED_DIR, DATA_DIR

# name, script, minutes, heavy, description
STAGES = [
    # name, script, minutes, heavy, description, required inputs
    ("typology", "04_edit_typology.py", 2, False, "pairwise edit measurements", []),
    ("embeddings", "05_embeddings.py", 6, True, "frozen BERT embeddings (needs torch)", []),
    ("main", "06_main_experiments.py", 12, True, "four tasks x four detectors", []),
    ("transfer", "07_transfer.py", 10, True, "leave-one-model/subfield/era protocols", []),
    ("robustness", "08_robustness.py", 6, True, "ablations, cross-era, non-identical fraction",
     ["results/pred_presence_base_ngr.npy"]),
    ("descriptives", "11_descriptives.py", 1, False, "Tables 4-6, variance decomposition",
     ["processed/pairs.pkl"]),
    ("recalibration", "14_recalibration.py", 3, False, "operating points, dose-response", []),
    ("magnitude", "15_magnitude_control.py", 3, False,
     "attribution under magnitude control", ["results/pred_type_base_ngr.npy"]),
    ("subfield", "16_subfield_validation.py", 1, False, "annotation agreement", []),
    ("method-probe", "../exploratory/18_method_probe.py", 3, True,
     "joint lattice and hierarchical comparators", []),
    ("palcp", "21_palcp.py", 2, False, "set-valued audit method",
     ["results/method_probe.json", "results/main_results_base.csv"]),
    ("audit", "40_isolation_audit.py", 1, False, "source-isolation audit", []),
    ("verify", "41_conformal_verification.py", 10, True, "coverage verification", []),
    ("operating", "42_operating_points.py", 3, False, "1% operating point, ablations", []),
    ("figures", "12_figures.py", 1, False, "Figures 1-7",
     ["processed/pairs.pkl", "results/descriptives.json", "results/transfer_results.csv",
      "results/pred_type_base_ngr.npy", "results/dose_response.csv",
      "results/per_condition_ceiling.csv", "results/per_type_model_detectability.csv",
      "results/attribution_by_magnitude.csv", "results/attribution_confusion.csv",
      "results/model_controlled.json", "results/cross_model_auc.csv",
      "results/transfer_baselines.csv", "results/paired_discrimination.csv",
      "results/fpr_strata.csv"]),
    ("figure-palcp", "22_figure_palcp.py", 1, False, "Figure 8",
     ["results/palcp_test.npz"]),
    ("figure-coverage", "44_figure_coverage.py", 1, False, "Figure 9",
     ["results/palcp_verification.json"]),
    ("tables", "13_tables.py", 1, False, "Markdown tables",
     ["results/main_results_base.csv", "results/main_results_bert.csv",
      "results/transfer_results.csv", "results/transfer_baselines.csv",
      "results/cross_model_auc.csv"]),
]

CORE = ["typology", "descriptives", "recalibration", "magnitude", "subfield", "method-probe", "palcp",
        "audit", "operating", "figures", "figure-palcp", "figure-coverage", "tables"]


def show(stages):
    print(f"{'stage':18s} {'min':>4s}  {'heavy':5s}  description")
    for name, _, mins, heavy, desc, _req in stages:
        print(f"{name:18s} {mins:4d}  {'yes' if heavy else '-':5s}  {desc}")


def select(args):
    names = [s[0] for s in STAGES]
    if args.stage == "all":
        chosen = names
    elif args.stage == "core":
        chosen = CORE
    else:
        chosen = names
    if args.from_stage:
        chosen = chosen[chosen.index(args.from_stage):]
    if args.only:
        wanted = [x.strip() for x in args.only.split(",")]
        unknown = [w for w in wanted if w not in names]
        if unknown:
            sys.exit(f"unknown stage(s): {unknown}")
        chosen = [n for n in names if n in wanted]
    return [s for s in STAGES if s[0] in chosen and not (args.skip_heavy and s[3])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--stage", default="core", choices=["core", "all"])
    ap.add_argument("--only", help="comma-separated stage names")
    ap.add_argument("--from", dest="from_stage", help="start at this stage")
    ap.add_argument("--skip-heavy", action="store_true", help="skip model-fitting stages")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failure")
    args = ap.parse_args()

    if args.list:
        show(STAGES)
        return 0

    stages = select(args)
    if not stages:
        print("no stages selected")
        return 0

    print(f"running {len(stages)} stage(s); estimated "
          f"{sum(s[2] for s in stages)} min\n")
    t0 = time.time()
    failures = []
    for name, script, _, _, desc, requires in stages:
        def input_path(relative):
            category, _, tail = relative.partition("/")
            base = {"results": RESULTS_DIR, "processed": PROCESSED_DIR,
                    "data": DATA_DIR}.get(category, ROOT / category)
            return base / tail
        missing = [r for r in requires if not input_path(r).exists()]
        if missing:
            print(f"[{name}] SKIPPED: missing {missing}. Run the producing stage first "
                  f"(see docs/reproduction.md).")
            failures.append(name)
            if not args.keep_going:
                break
            continue
        path = ROOT / "analysis" / script
        print(f"[{name}] {desc} ...", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, str(path)], cwd=ROOT)
        dt = time.time() - t
        if r.returncode != 0:
            failures.append(name)
            print(f"[{name}] FAILED after {dt:.0f}s", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"[{name}] ok ({dt:.0f}s)", flush=True)

    print(f"\ntotal {time.time() - t0:.0f}s")
    if failures:
        print("failed stages:", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
