"""Reconcile two reporting inconsistencies:

(A) The 1% false-positive operating point: ROC-interpolated values reported in Table 7
    versus the deployed threshold fixed on the calibration partition. Report both, plus
    the realised test FPR at the 1% threshold, per condition.

(B) The ablation table: recompute every variant under exactly the protocol used in the
    main detector table (dev-tuned C, balanced class weights) so that the "char n-gram
    only" row reproduces the headline AUC, and tabulate the difference against the
    earlier fixed-C unweighted run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import exp_core as ec

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
S = ec.prepare_stylo(df)
split = df.group_split.values
itr = np.where(split == "train")[0]
iva = np.where(split == "dev")[0]
icb = np.where(split == "calibration")[0]
ite = np.where(split == "test")[0]
y = df.intervention_present.values
OPS = ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


# ---------------- (A) deployed 1% operating point ----------------
print("=== (A) deployed operating points ===")
rows = []
for name, C in [("ngr", 4.0)]:
    det = ec.Detector("ngr", C=C, balanced=True).fit(
        df.abstract_clean.values[itr].tolist(), y[itr], S[itr])
    p_cb, _ = det.predict_proba(df.abstract_clean.values[icb].tolist(), S[icb])
    p_te, _ = det.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
    h_cb = p_cb[y[icb] == 0]
    for tag, q in [("5%", 0.95), ("1%", 0.99)]:
        thr = float(np.quantile(h_cb, q))
        flag = p_te >= thr
        fpr = float(flag[y[ite] == 0].mean())
        k = int(flag[y[ite] == 0].sum())
        rows.append(dict(operating_point=tag, threshold=thr,
                         calibration_fpr=float((h_cb >= thr).mean()),
                         test_fpr=fpr, test_fpr_lo=wilson(k, int((y[ite] == 0).sum()))[0],
                         test_fpr_hi=wilson(k, int((y[ite] == 0).sum()))[1],
                         test_tpr=float(flag[y[ite] == 1].mean()),
                         test_tpr_lo=wilson(int(flag[y[ite] == 1].sum()), int((y[ite] == 1).sum()))[0],
                         test_tpr_hi=wilson(int(flag[y[ite] == 1].sum()), int((y[ite] == 1).sum()))[1]))
        print(f"  deployed {tag}: thr={thr:.4f}  test FPR={fpr:.3f}  "
              f"test TPR={flag[y[ite]==1].mean():.3f}")

OP = pd.DataFrame(rows)

# per-condition TPR at both deployed points
rows = []
for tag, q in [("5%", 0.95), ("1%", 0.99)]:
    thr = float(np.quantile(p_cb[y[icb] == 0], q))
    flag = p_te >= thr
    op_te = df.operation.values[ite]
    for cond in ["human"] + OPS:
        m = (op_te == "original") if cond == "human" else (op_te == cond)
        k = int(flag[m].sum())
        rows.append(dict(operating_point=tag, condition=cond, n=int(m.sum()),
                         rate=float(flag[m].mean()),
                         lo=wilson(k, int(m.sum()))[0], hi=wilson(k, int(m.sum()))[1]))
PC = pd.DataFrame(rows)
print("\nper-condition rates at deployed thresholds:")
print(PC.pivot_table(index="condition", columns="operating_point", values="rate").round(3))

# ROC-referenced 1% values for comparison
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y[ite], p_te)
roc1 = float(np.interp(0.01, fpr, tpr))
roc5 = float(np.interp(0.05, fpr, tpr))
print(f"\nROC-referenced: TPR@1% = {roc1:.3f}, TPR@5% = {roc5:.3f}")
print(f"Deployed 1%: test FPR {OP.loc[OP.operating_point=='1%','test_fpr'].iloc[0]:.3f}, "
      f"TPR {OP.loc[OP.operating_point=='1%','test_tpr'].iloc[0]:.3f}")

OP.to_csv(os.path.join(OUT, "operating_points.csv"), index=False, encoding="utf-8-sig")
PC.to_csv(os.path.join(OUT, "operating_points_by_condition.csv"), index=False, encoding="utf-8-sig")

# ---------------- (B) ablation reconciliation ----------------
print("\n=== (B) ablation reconciliation ===")
LEN = ["log_chars", "n_sent", "mean_slen", "sd_slen", "cv_slen", "max_slen",
       "len_per_clause", "commas_per_sent"]
variants = [("char n-gram only", dict(kind="ngr")),
            ("stylometric only", dict(kind="sty")),
            ("stylometric, no length", dict(kind="sty", drop=LEN)),
            ("fused", dict(kind="hyb")),
            ("fused, no length", dict(kind="hyb", drop=LEN))]

GRID = [0.25, 1.0, 4.0, 16.0]
rows = []
for name, kw in variants:
    best, bestC = -1, None
    for C in GRID:
        d = ec.Detector(C=C, balanced=True, **kw).fit(
            df.abstract_clean.values[itr].tolist(), y[itr], S[itr])
        pv, _ = d.predict_proba(df.abstract_clean.values[iva].tolist(), S[iva])
        s = roc_auc_score(y[iva], pv)
        if s > best:
            best, bestC = s, C
    d = ec.Detector(C=bestC, balanced=True, **kw).fit(
        df.abstract_clean.values[itr].tolist(), y[itr], S[itr])
    p1, _ = d.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
    # fixed-C unweighted replication of the earlier table
    d0 = ec.Detector(C=4.0, balanced=False, **kw).fit(
        df.abstract_clean.values[itr].tolist(), y[itr], S[itr])
    p0, _ = d0.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
    rows.append(dict(variant=name, C_tuned=bestC,
                     auc_matched=float(roc_auc_score(y[ite], p1)),
                     tpr5_matched=float(ec.tpr_at_fpr(y[ite], p1, 0.05)),
                     auc_fixedC_unweighted=float(roc_auc_score(y[ite], p0)),
                     tpr5_fixedC_unweighted=float(ec.tpr_at_fpr(y[ite], p0, 0.05))))
    print(f"  {name:24s} C={bestC:<5g} matched AUC {rows[-1]['auc_matched']:.3f} "
          f"(fixed-C unweighted {rows[-1]['auc_fixedC_unweighted']:.3f})")

# metadata-only control under the same protocol
subs = pd.get_dummies(df.subfield).values.astype(float)
meta = np.hstack([df[["char_count"]].values / 100.0, subs])
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(meta[itr], y[itr])
pm = clf.predict_proba(meta[ite])[:, 1]
rows.append(dict(variant="metadata-only control", C_tuned=np.nan,
                 auc_matched=float(roc_auc_score(y[ite], pm)),
                 tpr5_matched=float(ec.tpr_at_fpr(y[ite], pm, 0.05)),
                 auc_fixedC_unweighted=np.nan, tpr5_fixedC_unweighted=np.nan))
print(f"  metadata-only control    matched AUC {rows[-1]['auc_matched']:.3f}")

AB = pd.DataFrame(rows)
AB.to_csv(os.path.join(OUT, "ablation_reconciled.csv"), index=False, encoding="utf-8-sig")

res = dict(operating_points=OP.to_dict(orient="records"),
           roc_tpr_at_1pct=roc1, roc_tpr_at_5pct=roc5,
           ablation=AB.to_dict(orient="records"))
json.dump(res, open(os.path.join(OUT, "fpr_ablation.json"), "w", encoding="utf-8"),
          indent=1, default=float)
print("\nsaved operating_points.csv / ablation_reconciled.csv / fpr_ablation.json")
