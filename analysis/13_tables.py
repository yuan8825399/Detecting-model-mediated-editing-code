"""Emit manuscript tables as markdown from result files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json, ast
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

RES = RESULTS_DIR
base = pd.read_csv(os.path.join(RES, "main_results_base.csv"), encoding="utf-8-sig")
bert = pd.read_csv(os.path.join(RES, "main_results_bert.csv"), encoding="utf-8-sig")
M = pd.concat([base, bert], ignore_index=True)
M["detector"] = M.detector.map({"sty": "Stylometric", "ngr": "Char n-gram",
                                "hyb": "Fused", "bert": "BERT (frozen)"})


def parse_ci(s):
    if not isinstance(s, str):
        return None
    try:
        v = ast.literal_eval(s.strip())
        return (float(v[0]), float(v[1]))
    except Exception:
        return None


for col in ["auc_ci", "mf1_ci", "tpr5_ci"]:
    if col in M.columns:
        M[col] = M[col].map(parse_ci)

TASKNAME = {"presence": "T1 Presence (all)", "presence_ne": "T1 Presence (effective edits only)",
            "type": "T2 Edit type (4-class)", "type_ne": "T2 Edit type (effective only)",
            "level": "T3 Edit intensity (ordinal 0–4)", "model": "T4 Model attribution (4-class)"}


def ci(row, key):
    try:
        a, b = row[key]
        return f"[{a:.3f}, {b:.3f}]"
    except Exception:
        return ""


lines = ["| Task | Detector | C | AUC [95% CI] | TPR@5% FPR | TPR@1% FPR | Balanced acc. | macro-F1 [95% CI] | QWK |",
         "|---|---|---|---|---|---|---|---|---|"]
for task in ["presence", "presence_ne", "type", "type_ne", "level", "model"]:
    sub = M[M.task == task]
    for _, r in sub.iterrows():
        c = r.get("auc_ci")
        auc = f"{r.auc:.3f} [{c[0]:.3f}, {c[1]:.3f}]" if (pd.notna(r.get("auc")) and c) else (
            f"{r.auc:.3f}" if pd.notna(r.get("auc")) else "—")
        t5 = f"{r.tpr_at_5pct:.3f}" if pd.notna(r.get("tpr_at_5pct")) else "—"
        t1 = f"{r.tpr_at_1pct:.3f}" if pd.notna(r.get("tpr_at_1pct")) else "—"
        ba = f"{r.bacc:.3f}" if pd.notna(r.get("bacc")) else "—"
        cf = r.get("mf1_ci")
        mf = f"{r.mf1:.3f} [{cf[0]:.3f}, {cf[1]:.3f}]" if (pd.notna(r.get("mf1")) and cf) else (
            f"{r.mf1:.3f}" if pd.notna(r.get("mf1")) else "—")
        qwk = f"{r.qwk:.3f}" if pd.notna(r.get("qwk")) else "—"
        lines.append(f"| {TASKNAME[task]} | {r.detector} | {r.C:g} | {auc} | {t5} | {t1} | {ba} | {mf} | {qwk} |")
open(os.path.join(RES, "table4_detectors.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

# Table 5: reliability summary
tr = pd.read_csv(os.path.join(RES, "transfer_results.csv"), encoding="utf-8-sig")
lomo = tr[tr.protocol == "LOMO"]
loso = tr[tr.protocol == "LOSO"]
matched = pd.read_csv(RES / "transfer_baselines.csv").set_index("task")
rows = [("In-domain (fused, all models)", float(matched.loc["presence", "auc"]),
         float(matched.loc["type", "mf1"]), float(matched.loc["level", "qwk"]))]
l = lomo[lomo.task == "presence"]
rows.append(("Leave-one-model-out (mean of 4)", l.auc.mean(), np.nan, np.nan))
for _, r in lomo.iterrows():
    rows.append((f"  held-out {r.heldout}", r.auc, r.get("mf1") if r.task == "type" else np.nan, np.nan))
s = loso[loso.task == "presence"]
rows.append(("Leave-one-subfield-out (mean of 5)", s.auc.mean(), np.nan, np.nan))
for _, r in loso[loso.task == "type"].iterrows():
    pass
ci_ = tr[tr.protocol == "crossintensity"]
for _, r in ci_.iterrows():
    rows.append((f"Cross-intensity {r.train_on} → {r.test_on}", r.auc, np.nan, np.nan))

lines = ["| Protocol | Presence AUC |", "|---|---|"]
for a, b, c, d in rows:
    if isinstance(b, float) and not np.isnan(b):
        lines.append(f"| {a} | {b:.3f} |")
open(os.path.join(RES, "table5_reliability.md"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

print("\nLOMO type mf1:", round(lomo[lomo.task == "type"].mf1.mean(), 3),
      "range", round(lomo[lomo.task == "type"].mf1.min(), 3), round(lomo[lomo.task == "type"].mf1.max(), 3))
print("LOMO level qwk:", round(lomo[lomo.task == "level"].qwk.mean(), 3),
      "range", round(lomo[lomo.task == "level"].qwk.min(), 3), round(lomo[lomo.task == "level"].qwk.max(), 3))
print("LOSO type mf1:", round(loso[loso.task == "type"].mf1.mean(), 3),
      round(loso[loso.task == "type"].mf1.min(), 3), round(loso[loso.task == "type"].mf1.max(), 3))
print("LOSO level qwk:", round(loso[loso.task == "level"].qwk.mean(), 3),
      round(loso[loso.task == "level"].qwk.min(), 3), round(loso[loso.task == "level"].qwk.max(), 3))
print("LOSO presence:", loso[loso.task == "presence"].auc.round(3).tolist())
CM = pd.read_csv(os.path.join(RES, "cross_model_auc.csv"), index_col=0, encoding="utf-8-sig")
v = CM.values.astype(float)
print("cross-model diag mean", np.diag(v).mean().round(3), "offdiag mean", v[~np.eye(4, dtype=bool)].mean().round(3),
      "min", v.min().round(3))
