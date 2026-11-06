"""Conformal regression intervals for edit depth (text-only), plus diagnostics of
where the presence method's decisive 'absent' verdicts fail."""

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
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import feats

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
split = df.group_split.values
itr = np.where(split == "train")[0]
iva = np.where(split == "calibration")[0]
ite = np.where(split == "test")[0]

OPS = ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]
MODS = ["deepseek", "doubao", "kimi", "wenxin"]

vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                      sublinear_tf=True, max_features=300000)
Xtr = vec.fit_transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[itr]])
Xva = vec.transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[iva]])
Xte = vec.transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[ite]])

g_train = df.normalized_edit_distance.values[itr]
g_cal = df.normalized_edit_distance.values[iva]
g_te = df.normalized_edit_distance.values[ite]

reg = Ridge(alpha=1.0, solver="lsqr").fit(Xtr, g_train)
gh = reg.predict(Xva); gt = reg.predict(Xte)

res_cal = np.abs(g_cal - gh)
alpha = 0.10
n = len(res_cal)
k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
q = float(np.sort(res_cal)[k - 1])
lo = np.clip(gt - q, 0, 1.6); hi = gt + q
cov = (g_te >= lo) & (g_te <= hi)
print(f"conformal regression: q = {q:.3f}, coverage = {cov.mean():.3f}, "
      f"mean width = {(hi-lo).mean():.3f}")
print("coverage by true depth bin:")
bins = [-1e-9, 1e-9, 0.05, 0.15, 0.30, 10]
labs = ["0", "(0,0.05]", "(0.05,0.15]", "(0.15,0.30]", ">0.30"]
b = np.asarray(pd.cut(g_te, bins, labels=labs).astype(str))
rows = []
for lab in labs:
    msk = b == lab
    if not msk.sum():
        continue
    rows.append(dict(bin=lab, n=int(msk.sum()), coverage=float(cov[msk].mean()),
                     width=float((hi - lo)[msk].mean()), mean_ghat=float(gt[msk].mean())))
    print(f"  {lab:12s} n={msk.sum():4d} cov={cov[msk].mean():.2f} width={(hi-lo)[msk].mean():.2f} "
          f"ghat={gt[msk].mean():.2f}")
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "depth_intervals.csv"), index=False, encoding="utf-8-sig")

# ---- where do decisive-absent verdicts fail? ----
prov = np.array([f"{m}|{o}" for m, o in zip(df.provider.values, df.operation.values)])
prov[df.intervention_present.values == 0] = "H0"
clf = LogisticRegression(C=4.0, max_iter=3000, solver="lbfgs").fit(Xtr, prov[itr])
Pva = clf.predict_proba(Xva); Pte = clf.predict_proba(Xte)
h0 = list(clf.classes_).index("H0")
pres_va = 1 - Pva[:, h0]; pres_te = 1 - Pte[:, h0]

y = df.intervention_present.values
s_va = np.where(y[iva] == 1, 1 - pres_va, pres_va)
def qh(s, a):
    kk = min(int(np.ceil((len(s) + 1) * (1 - a))), len(s))
    return float(np.sort(s)[kk - 1])
q_pos = qh(s_va[y[iva] == 1], alpha); q_neg = qh(s_va[y[iva] == 0], alpha)
inc_p = (1 - pres_te) <= q_pos + 1e-12
inc_a = pres_te <= q_neg + 1e-12
dec_a = (inc_a & ~inc_p)

wrong = dec_a & (y[ite] == 1)
print(f"\ndecisive-absent: {int(dec_a.sum())}, of which model-processed: {int(wrong.sum())}")
comp = pd.Series(df.operation.values[ite][wrong]).value_counts()
print("composition of false-absent verdicts by true operation:")
print(comp.to_string())
print("share that are grammar-correction:",
      round(float((df.operation.values[ite][wrong] == "grammar_correction").mean()), 3))
print("share of all grammar-correction test records that receive a decisive-absent verdict:",
      round(float(dec_a[df.operation.values[ite] == "grammar_correction"].mean()), 3))
print("same for human originals:", round(float(dec_a[y[ite] == 0].mean()), 3))

json.dump(dict(q=q, coverage=float(cov.mean()), width=float((hi - lo).mean()),
               decisive_absent=int(dec_a.sum()), false_absent=int(wrong.sum()),
               share_E1_of_false=float((df.operation.values[ite][wrong] == "grammar_correction").mean())),
          open(os.path.join(OUT, "depth_ci.json"), "w", encoding="utf-8"), indent=1, default=float)
print("\nsaved")
