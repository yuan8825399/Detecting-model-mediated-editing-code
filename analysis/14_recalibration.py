"""Recompute all operating-point results with thresholds fixed on the CALIBRATION
partition (never on test). Also: per-condition TPR vs detectability ceiling,
Wilson intervals per dose bin, and a rank-correlation trend test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import exp_core as ec

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
S = ec.prepare_stylo(df)
split = df.group_split.values

itr = np.where(split == "train")[0]
iva = np.where(split == "calibration")[0]
ite = np.where(split == "test")[0]

OPS = ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]
MODELS = ["deepseek", "doubao", "kimi", "wenxin"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


# ---------------- fit once on train ----------------
det_bal = ec.Detector("ngr", C=4.0, balanced=True)
det_bal.fit(df.abstract_clean.values[itr].tolist(), df.intervention_present.values[itr], S[itr])
det_unc = ec.Detector("ngr", C=4.0, balanced=False)
det_unc.fit(df.abstract_clean.values[itr].tolist(), df.intervention_present.values[itr], S[itr])

p_cal, _ = det_bal.predict_proba(df.abstract_clean.values[iva].tolist(), S[iva])
p_cal_u, _ = det_unc.predict_proba(df.abstract_clean.values[iva].tolist(), S[iva])
p_te, _ = det_bal.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
p_te_u, _ = det_unc.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])

y_cal = df.intervention_present.values[iva]
y_te = df.intervention_present.values[ite]
orig_cal = df.iloc[iva][df.iloc[iva].operation.values == "original"]
orig_cal_p = p_cal[orig_cal.index.map({i: k for k, i in enumerate(iva)})] if False else None

# human-only scores on calibration partition
mask_cal_orig = df.operation.values[iva] == "original"
mask_te_orig = df.operation.values[ite] == "original"
hp_cal = p_cal[mask_cal_orig]
hp_te = p_te[mask_te_orig]
print("calibration originals:", mask_cal_orig.sum(), " test originals:", mask_te_orig.sum())

thr5 = float(np.quantile(hp_cal, 0.95))
thr1 = float(np.quantile(hp_cal, 0.99))
cal_fpr5 = float((hp_cal >= thr5).mean())
cal_fpr1 = float((hp_cal >= thr1).mean())
print(f"threshold@5% (calibration) = {thr5:.4f}; realised calibration FPR = {cal_fpr5:.3f}")
print(f"threshold@1% (calibration) = {thr1:.4f}; realised calibration FPR = {cal_fpr1:.3f}")

R = dict(thr5=thr5, thr1=thr1, cal_fpr5=cal_fpr5, cal_fpr1=cal_fpr1,
         n_cal_orig=int(mask_cal_orig.sum()), n_test_orig=int(mask_te_orig.sum()))

# ---------------- test-set operating characteristics ----------------
te = df.iloc[ite].reset_index(drop=True)
te["p"] = p_te
te["p_u"] = p_te_u
te["flag5"] = (te.p >= thr5).astype(int)
te["flag1"] = (te.p >= thr1).astype(int)
orig_te = te[te.operation == "original"]
ed_te = te[te.operation != "original"].copy()

fpr_te = float((orig_te.p >= thr5).mean())
R["test_fpr5"] = fpr_te
R["test_fpr5_ci"] = wilson(int((orig_te.p >= thr5).sum()), len(orig_te))
R["overall_tpr5"] = float(ed_te.flag5.mean())
R["overall_tpr5_ci"] = wilson(int(ed_te.flag5.sum()), len(ed_te))
print(f"test FPR@cal-threshold = {fpr_te:.3f}; overall TPR = {R['overall_tpr5']:.3f}")

# ---------------- per-condition TPR vs ceiling ----------------
rows = []
for op in OPS:
    e = ed_te[ed_te.operation == op]
    ident = float((e.identical == 1).mean())
    ceil = 1 - ident
    k = int(e.flag5.sum()); n = len(e)
    rows.append(dict(operation=op, n=n, identical_rate=ident, ceiling=ceil,
                     tpr=float(e.flag5.mean()), tpr_lo=wilson(k, n)[0], tpr_hi=wilson(k, n)[1],
                     tpr_at_1pct=float(e.flag1.mean()),
                     normalised_tpr=float(e.flag5.mean() / ceil) if ceil > 0 else np.nan,
                     mean_ed=float(e.normalized_edit_distance.mean())))
PC = pd.DataFrame(rows)
# pooled weighted ceiling
pooled_ceil = float((PC.ceiling * PC.n).sum() / PC.n.sum())
PC_pooled = dict(operation="pooled", n=int(PC.n.sum()), identical_rate=float(1 - pooled_ceil),
                 ceiling=pooled_ceil, tpr=R["overall_tpr5"], tpr_lo=R["overall_tpr5_ci"][0],
                 tpr_hi=R["overall_tpr5_ci"][1], tpr_at_1pct=float(ed_te.flag1.mean()),
                 normalised_tpr=R["overall_tpr5"] / pooled_ceil,
                 mean_ed=float(ed_te.normalized_edit_distance.mean()))
PC = pd.concat([PC, pd.DataFrame([PC_pooled])], ignore_index=True)
print("\n=== per-condition TPR vs ceiling ===")
print(PC.round(3).to_string(index=False))
PC.to_csv(os.path.join(OUT, "per_condition_ceiling.csv"), index=False, encoding="utf-8-sig")
R["pooled_ceiling"] = pooled_ceil

# ---------------- dose-response, merged bins + Wilson CI ----------------
bins = [-1e-9, 1e-9, 0.05, 0.10, 0.20, 0.30, 0.40, 10]
labels = ["0", "(0, 0.05]", "(0.05, 0.10]", "(0.10, 0.20]", "(0.20, 0.30]",
          "(0.30, 0.40]", "> 0.40"]
ed_te["mag_bin"] = pd.cut(ed_te.normalized_edit_distance, bins=bins, labels=labels, right=True)
rows = []
for lab in labels:
    g = ed_te[ed_te.mag_bin == lab]
    if len(g) == 0:
        continue
    k = int(g.flag5.sum()); n = len(g)
    lo, hi = wilson(k, n)
    rows.append(dict(bin=lab, n=n, mean_ed=float(g.normalized_edit_distance.mean()),
                     tpr=float(g.flag5.mean()), lo=lo, hi=hi,
                     tpr_at_1pct=float(g.flag1.mean())))
DOSE = pd.DataFrame(rows)
print("\n=== dose-response (merged bins, Wilson 95% CI) ===")
print(DOSE.round(3).to_string(index=False))
DOSE.to_csv(os.path.join(OUT, "dose_response.csv"), index=False, encoding="utf-8-sig")

# trend tests
rho, p_rho = stats.spearmanr(ed_te.normalized_edit_distance, ed_te.p)
R["spearman_rho"] = float(rho); R["spearman_p"] = float(p_rho)
print(f"\nSpearman rho(magnitude, score) = {rho:.3f}, p = {p_rho:.2e}")

# Cochran-Armitage style trend on binned detection (score test)
n_b = DOSE.n.values.astype(float)
y_b = (DOSE.tpr.values * n_b)
x_b = np.arange(len(n_b)).astype(float)
pbar = y_b.sum() / n_b.sum()
num = np.sum(x_b * (y_b - n_b * pbar))
den = np.sqrt(pbar * (1 - pbar) * (np.sum(n_b * x_b ** 2) - (np.sum(n_b * x_b)) ** 2 / n_b.sum()))
z_ca = num / den
R["ca_z"] = float(z_ca)
R["ca_p"] = float(2 * (1 - stats.norm.cdf(abs(z_ca))))
print(f"Cochran-Armitage trend z = {z_ca:.1f}, p = {R['ca_p']:.2e}")

# document-clustered bootstrap CI for Spearman rho
rng = np.random.default_rng(0)
gidx = {g: np.where(ed_te.source_id.values == g)[0] for g in ed_te.source_id.unique()}
ug = list(gidx)
boot = []
for _ in range(1000):
    idx = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug), replace=True)])
    try:
        boot.append(stats.spearmanr(ed_te.normalized_edit_distance.values[idx],
                                    ed_te.p.values[idx]).statistic)
    except Exception:
        pass
R["spearman_ci"] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
print("rho CI:", R["spearman_ci"])

# ---------------- per type x model ----------------
rows = []
for op in OPS:
    for m in MODELS:
        g = ed_te[(ed_te.operation == op) & (ed_te.provider == m)]
        k = int(g.flag5.sum()); n = len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(operation=op, provider=m, n=n, tpr=float(g.flag5.mean()),
                         lo=lo, hi=hi, mean_ed=float(g.normalized_edit_distance.mean())))
TM = pd.DataFrame(rows)
TM.to_csv(os.path.join(OUT, "per_type_model_detectability.csv"), index=False, encoding="utf-8-sig")
print("\n=== TPR by type x model ===")
print(TM.pivot_table(index="operation", columns="provider", values="tpr").round(3))

# ---------------- FPR on human text by subfield / era ----------------
rows = []
for key in ["era", "subfield"]:
    for v, g in orig_te.groupby(key):
        k = int((g.p >= thr5).sum()); n = len(g)
        lo, hi = wilson(k, n)
        rows.append(dict(stratum=key, level=str(v), n=n, fpr=float(k / n), lo=lo, hi=hi,
                         mean_p=float(g.p.mean())))
FPR = pd.DataFrame(rows)
FPR.to_csv(os.path.join(OUT, "fpr_strata.csv"), index=False, encoding="utf-8-sig")
print("\n=== FPR on human text ===")
print(FPR.round(3).to_string(index=False))

# ---------------- paired discrimination ----------------
o = orig_te.set_index("source_id").p
pair = []
for (sid, opn), g in ed_te.groupby(["source_id", "operation"]):
    if sid not in o.index:
        continue
    for _, r in g.iterrows():
        pair.append((sid, opn, r.provider, float(o[sid]), float(r.p), float(r.p - o[sid])))
PR = pd.DataFrame(pair, columns=["source_id", "operation", "provider", "p_orig", "p_edit", "delta"])
PR["win"] = (PR.delta > 0).astype(int)
rows = []
for op in OPS:
    for m in MODELS:
        g = PR[(PR.operation == op) & (PR.provider == m)]
        k = int(g.win.sum()); n = len(g)
        rows.append(dict(operation=op, provider=m, n=n, win=float(k / n),
                         lo=wilson(k, n)[0], hi=wilson(k, n)[1]))
PW = pd.DataFrame(rows)
PW.to_csv(os.path.join(OUT, "paired_discrimination.csv"), index=False, encoding="utf-8-sig")
R["paired_overall"] = float(PR.win.mean())
print("\npaired overall:", round(R["paired_overall"], 3))
print(PW.pivot_table(index="operation", columns="provider", values="win").round(3))

# ---------------- calibration metrics on test (unweighted model) ----------------
ece = ec.ece(y_te, p_te_u)
brier = float(np.mean((p_te_u - y_te) ** 2))
R["ece_test"] = float(ece); R["brier_test"] = brier
R["mean_p_test"] = float(p_te_u.mean()); R["obs_test"] = float(y_te.mean())
rel = []
edges = np.linspace(0, 1, 11)
idx = np.clip(np.digitize(p_te_u, edges[1:-1]), 0, 9)
for b in range(10):
    m = idx == b
    if m.sum() == 0:
        continue
    k = int(y_te[m].sum()); n = int(m.sum())
    rel.append(dict(bin=f"{edges[b]:.1f}-{edges[b+1]:.1f}", n=n, mean_p=float(p_te_u[m].mean()),
                    obs=float(y_te[m].mean()), lo=wilson(k, n)[0], hi=wilson(k, n)[1]))
REL = pd.DataFrame(rel)
REL.to_csv(os.path.join(OUT, "reliability.csv"), index=False, encoding="utf-8-sig")
print("\n=== reliability (test) ===")
print(REL.round(3).to_string(index=False))
print(f"ECE={ece:.4f} Brier={brier:.4f} mean_p={p_te_u.mean():.4f} obs={y_te.mean():.4f}")

# Brier skill score vs constant base rate
bs_ref = float(np.mean((y_te.mean() - y_te) ** 2))
R["brier_skill"] = float(1 - brier / bs_ref)
print("Brier skill score vs base rate:", round(R["brier_skill"], 4))

json.dump(R, open(os.path.join(OUT, "recalib.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\nsaved recalib.json")
