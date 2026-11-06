"""Prototype: ceiling-aware, magnitude-conditioned conformal provenance sets (CAPS).

Idea: instead of a binary verdict, return a *set* of plausible provenance labels with
distribution-free coverage, where the calibration stratum is chosen by a text-only
estimate of edit depth. Tests whether (a) marginal conformal coverage hides
class-conditional failure on undetectable edits, and (b) magnitude-conditioned
(Mondrian) calibration repairs it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import exp_core as ec
from provenance import feats

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
S = ec.prepare_stylo(df)
split = df.group_split.values
LEVELS = ["original", "grammar_correction", "academic_polish",
          "sentence_restructure", "abstract_rewrite"]
y_all = df.operation.values

itr = np.where(split == "train")[0]
iva = np.where(split == "calibration")[0]
ite = np.where(split == "test")[0]

# ---------------- base 5-class model ----------------
vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                      sublinear_tf=True, max_features=300000)
Xtr = vec.fit_transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[itr]])
Xva = vec.transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[iva]])
Xte = vec.transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[ite]])

clf = LogisticRegression(C=0.25, max_iter=3000, solver="lbfgs")
clf.fit(Xtr, y_all[itr])
classes = clf.classes_
print("classes:", classes)

Pva = clf.predict_proba(Xva)
Pte = clf.predict_proba(Xte)
yte = y_all[ite]

# ---------------- text-only edit-depth regressor ----------------
reg = Ridge(alpha=1.0, solver="lsqr")
reg.fit(Xtr, df.normalized_edit_distance.values[itr])
ghat_va = reg.predict(Xva)
ghat_te = reg.predict(Xte)
g_te = df.normalized_edit_distance.values[ite]
rho = stats.spearmanr(ghat_te, g_te).statistic
r2 = 1 - np.sum((g_te - ghat_te) ** 2) / np.sum((g_te - g_te.mean()) ** 2)
print(f"magnitude regressor: test R2 = {r2:.3f}, Spearman = {rho:.3f}")

# ---------------- conformal machinery ----------------
def conformal_scores(P, y):
    idx = np.array([list(classes).index(v) for v in y])
    return 1 - P[np.arange(len(y)), idx]


def qhat(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return np.sort(scores)[k - 1]


def build_sets(P, q, ):
    s = 1 - P
    return s <= q + 1e-12


def evaluate(setmask, y, name, alpha=0.1):
    idx = np.array([list(classes).index(v) for v in y])
    covered = setmask[np.arange(len(y)), idx]
    size = setmask.sum(1)
    singleton = size == 1
    pred_single = np.array([classes[i] for i in setmask.argmax(1)])
    acc_single = (pred_single[singleton] == y[singleton]).mean() if singleton.any() else np.nan
    out = dict(variant=name, coverage=float(covered.mean()), mean_size=float(size.mean()),
               singleton_rate=float(singleton.mean()), singleton_acc=float(acc_single),
               median_size=float(np.median(size)))
    out["cov_by_class"] = {c: float(covered[y == c].mean()) for c in classes if (y == c).any()}
    return out


alpha = 0.10
results = []

# (1) naive split conformal
s_va = conformal_scores(Pva, y_all[iva])
q_glob = qhat(s_va, alpha)
results.append(evaluate(build_sets(Pte, q_glob), yte, "Split conformal (global)"))

# (2) Mondrian by TRUE edit magnitude (oracle covariate)
TRUE_BINS = [-1e-9, 1e-9, 0.05, 0.15, 0.30, 10]
LABS = ["0", "(0,0.05]", "(0.05,0.15]", "(0.15,0.30]", ">0.30"]
gva_true = df.normalized_edit_distance.values[iva]
b_va_true = np.asarray(pd.cut(gva_true, TRUE_BINS, labels=LABS).astype(str))
b_te_true = np.asarray(pd.cut(g_te, TRUE_BINS, labels=LABS).astype(str))
qmap = {}
for lab in LABS:
    m = b_va_true == lab
    qmap[lab] = qhat(s_va[m], alpha) if m.sum() >= 20 else q_glob
mask = np.zeros_like(Pte, dtype=bool)
for lab in LABS:
    rows = b_te_true == lab
    if rows.sum():
        mask[rows] = build_sets(Pte[rows], qmap[lab])
results.append(evaluate(mask, yte, "Mondrian, true magnitude (oracle)"))

# (3) Mondrian by PREDICTED edit magnitude (deployable)
qcuts = np.quantile(np.clip(ghat_va, 0, None), [0.25, 0.5, 0.75])
def pred_bin(g):
    return np.digitize(np.clip(g, 0, None), qcuts)
b_va_p = pred_bin(ghat_va)
b_te_p = pred_bin(ghat_te)
qmap_p = {}
for k in range(4):
    m = b_va_p == k
    qmap_p[k] = qhat(s_va[m], alpha) if m.sum() >= 20 else q_glob
mask = np.zeros_like(Pte, dtype=bool)
for k in range(4):
    rows = b_te_p == k
    if rows.sum():
        mask[rows] = build_sets(Pte[rows], qmap_p[k])
results.append(evaluate(mask, yte, "Mondrian, predicted magnitude"))

# (4) class-conditional (Mondrian by true class) - standard alternative, uses labels at calib
qmap_c = {c: qhat(s_va[y_all[iva] == c], alpha) for c in classes}
mask = np.zeros_like(Pte, dtype=bool)
for c in classes:
    rows = yte == c
    if rows.sum():
        mask[rows] = build_sets(Pte[rows], qmap_c[c])
results.append(evaluate(mask, yte, "Class-conditional (oracle class)"))

R = pd.DataFrame(results)
print("\n=== conformal variants (alpha = 0.10) ===")
print(R[["variant", "coverage", "mean_size", "median_size", "singleton_rate", "singleton_acc"]].round(3).to_string(index=False))
print("\ncoverage by true class:")
for _, r in R.iterrows():
    print(f"  {r.variant:36s}", {k: round(v, 2) for k, v in r.cov_by_class.items()})

# ---------------- set composition ----------------
mask_glob = build_sets(Pte, q_glob)
print("\nshare of rows whose set contains all five labels (global):",
      round(float(mask_glob.all(1).mean()), 3))
print("mean set size by true class (global):")
for c in classes:
    print(f"  {c:22s} {mask_glob[yte==c].sum(1).mean():.2f}")
print("\nmean set size by true magnitude bin (global):")
for lab in LABS:
    rows = b_te_true == lab
    if rows.sum():
        print(f"  {lab:12s} n={rows.sum():4d}  size={mask_glob[rows].sum(1).mean():.2f}"
              f"  E1-in-set={(mask_glob[rows][:, list(classes).index('grammar_correction')]).mean():.2f}"
              f"  H0-in-set={(mask_glob[rows][:, list(classes).index('original')]).mean():.2f}")

json.dump(dict(r2=r2, rho=float(rho), alpha=alpha,
               variants=results), open(os.path.join(OUT, "conformal_proto.json"), "w",
                                       encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
R.to_csv(os.path.join(OUT, "conformal_proto.csv"), index=False, encoding="utf-8-sig")
print("\nsaved")
