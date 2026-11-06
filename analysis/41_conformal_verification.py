"""PAL-CP calibration and uncertainty verification.

(1) Coverage vs nominal level for class-conditional and marginal conformal.
(2) Split-robustness: repeat the whole train/calibration/test pipeline over random
    document-level partitions and compare achieved coverage to the nominal level.
(3) Exchangeability stress test: calibrate on three models, test on the held-out model.
(4) Document-level bootstrap intervals for coverage, decisiveness and verdict precision.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import feats
from provenance.conformal import quantile_nonconformity
from provenance.exp_core import group_bootstrap

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
texts = df.abstract_clean.values
split = df.group_split.values
y = df.intervention_present.values
src = df.source_id.values
prov = df.provider.values
prov = np.array(["H0" if p == "" else p for p in prov])
OPS = ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]
MODS = ["deepseek", "doubao", "kimi", "wenxin"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def qhat(scores, alpha):
    return quantile_nonconformity(scores, alpha)


class Pipeline:
    def __init__(self, C=4.0):
        self.C = C

    def fit(self, tr_idx):
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                                   sublinear_tf=True, max_features=300000)
        Xtr = self.vec.fit_transform([feats.clean_for_ngram(texts[i]) for i in tr_idx])
        lab = np.array([f"{prov[i]}|{df.operation.values[i]}" for i in tr_idx])
        lab[~y[tr_idx].astype(bool)] = "H0"
        self.clf = LogisticRegression(C=self.C, max_iter=3000, solver="lbfgs").fit(Xtr, lab)
        self.h0 = list(self.clf.classes_).index("H0")
        return self

    def presence(self, idx):
        X = self.vec.transform([feats.clean_for_ngram(texts[i]) for i in idx])
        return 1 - self.clf.predict_proba(X)[:, self.h0]


def conformal_eval(p_cal, y_cal, p_te, y_te, alpha, mondrian=True):
    s_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
    if mondrian:
        q_pos = qhat(s_cal[y_cal == 1], alpha)
        q_neg = qhat(s_cal[y_cal == 0], alpha)
    else:
        q = qhat(s_cal, alpha)
        q_pos = q_neg = q
    inc_p = (1 - p_te) <= q_pos + 1e-12
    inc_a = p_te <= q_neg + 1e-12
    cov = np.where(y_te == 1, inc_p, inc_a)
    dec = inc_p ^ inc_a
    return dict(coverage=float(cov.mean()),
                cov_edited=float(cov[y_te == 1].mean()) if (y_te == 1).any() else np.nan,
                cov_human=float(cov[y_te == 0].mean()) if (y_te == 0).any() else np.nan,
                decisive=float(dec.mean()),
                inc_p=inc_p, inc_a=inc_a, cov=cov, dec=dec)


# ---------------- (1) coverage vs nominal on the official split ----------------
itr = np.where(split == "train")[0]
icb = np.where(split == "calibration")[0]
ite = np.where(split == "test")[0]
P = Pipeline().fit(itr)
p_cb, p_te = P.presence(icb), P.presence(ite)
y_cb, y_te = y[icb], y[ite]

rows = []
for alpha in [0.01, 0.05, 0.10, 0.20, 0.30]:
    for mond in (True, False):
        r = conformal_eval(p_cb, y_cb, p_te, y_te, alpha, mond)
        k = int(r["cov"].sum())
        lo, hi = group_bootstrap(y_te, r["cov"], lambda y, c: c.mean(),
                                 src[ite], n_boot=2000, seed=7)
        rows.append(dict(alpha=alpha, nominal=1 - alpha,
                         variant="class-conditional" if mond else "marginal",
                         coverage=r["coverage"], lo=lo, hi=hi,
                         cov_edited=r["cov_edited"], cov_human=r["cov_human"],
                         decisive=r["decisive"],
                         covers_nominal=bool(lo <= 1 - alpha <= hi)))
C = pd.DataFrame(rows)
C.to_csv(os.path.join(OUT, "conformal_coverage_curve.csv"), index=False, encoding="utf-8-sig")
print("=== (1) coverage vs nominal, official split ===")
print(C.round(3).to_string(index=False))

# ---------------- (2) split robustness ----------------
print("\n=== (2) random document-level partitions (10 replicates) ===")
docs = np.unique(src)
rng = np.random.default_rng(2026)
rep = []
for b in range(10):
    perm = rng.permutation(docs)
    n = len(perm)
    tr_d = set(perm[:int(0.6 * n)])
    cb_d = set(perm[int(0.6 * n):int(0.75 * n)])
    te_d = set(perm[int(0.75 * n):])
    idx = {d: np.where(src == d)[0] for d in docs}
    tr_i = np.concatenate([idx[d] for d in sorted(tr_d)])
    cb_i = np.concatenate([idx[d] for d in sorted(cb_d)])
    te_i = np.concatenate([idx[d] for d in sorted(te_d)])
    Pp = Pipeline().fit(tr_i)
    pc, pt = Pp.presence(cb_i), Pp.presence(te_i)
    r = conformal_eval(pc, y[cb_i], pt, y[te_i], 0.10, True)
    rep.append(dict(rep=b, coverage=r["coverage"], cov_edited=r["cov_edited"],
                    cov_human=r["cov_human"], decisive=r["decisive"],
                    n_test=int(len(te_i))))
    print(f"  rep {b}: coverage {r['coverage']:.3f} (edited {r['cov_edited']:.3f}, "
          f"human {r['cov_human']:.3f}), decisive {r['decisive']:.3f}")
R = pd.DataFrame(rep)
R.to_csv(os.path.join(OUT, "conformal_split_robustness.csv"), index=False, encoding="utf-8-sig")
print(f"  mean coverage {R.coverage.mean():.3f} (sd {R.coverage.std():.3f}); "
      f"nominal 0.90; replicates below nominal: {int((R.coverage < 0.90).sum())}/10")

# ---------------- (3) exchangeability stress test ----------------
print("\n=== (3) calibration with a held-out editing model ===")
rows = []
for held in MODS:
    tr_i = np.where((split == "train") & (prov != held))[0]
    cb_i = np.where((split == "calibration") & (prov != held))[0]
    te_i = np.where((split == "test") & ((prov == held) | (prov == "H0")))[0]
    Px = Pipeline().fit(tr_i)
    pc, pt = Px.presence(cb_i), Px.presence(te_i)
    r = conformal_eval(pc, y[cb_i], pt, y[te_i], 0.10, True)
    rows.append(dict(heldout=held, coverage=r["coverage"], cov_edited=r["cov_edited"],
                     cov_human=r["cov_human"], decisive=r["decisive"], n_test=int(len(te_i))))
    print(f"  held-out {held}: coverage {r['coverage']:.3f} "
          f"(edited {r['cov_edited']:.3f}, human {r['cov_human']:.3f})")
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "conformal_shift.csv"), index=False, encoding="utf-8-sig")

# ---------------- (4) bootstrap intervals ----------------
print("\n=== (4) document-level bootstrap intervals (alpha = 0.10) ===")
r = conformal_eval(p_cb, y_cb, p_te, y_te, 0.10, True)
inc_p, inc_a, dec = r["inc_p"], r["inc_a"], r["dec"]
fa = dec & (y_te == 0)
dec_p = dec & inc_p
dec_a = dec & inc_a
te_docs = src[ite]
ug = np.unique(te_docs)
gidx = {g: np.where(te_docs == g)[0] for g in ug}
rng2 = np.random.default_rng(7)
samples = {k: [] for k in ["coverage", "decisive", "precision_present", "precision_absent"]}
for _ in range(2000):
    idx = np.concatenate([gidx[g] for g in rng2.choice(ug, len(ug), replace=True)])
    samples["coverage"].append(np.where(y_te[idx] == 1, inc_p[idx], inc_a[idx]).mean())
    samples["decisive"].append(dec[idx].mean())
    m = dec_p[idx]
    if m.any():
        samples["precision_present"].append(y_te[idx][m].mean())
    m = dec_a[idx]
    if m.any():
        samples["precision_absent"].append(1 - y_te[idx][m].mean())
boot = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in samples.items() if v}
point = dict(coverage=r["coverage"], decisive=r["decisive"],
             precision_present=float(y_te[dec_p].mean()),
             precision_absent=float(1 - y_te[dec_a].mean()))
for k in point:
    print(f"  {k:18s} {point[k]:.3f}  [{boot[k][0]:.3f}, {boot[k][1]:.3f}]")

res = dict(coverage_curve=C.to_dict(orient="records"),
           split_robustness=dict(mean=float(R.coverage.mean()), sd=float(R.coverage.std()),
                                 min=float(R.coverage.min()), max=float(R.coverage.max()),
                                 below_nominal=int((R.coverage < 0.90).sum())),
           shift=S.to_dict(orient="records"),
           bootstrap=dict(point=point, ci=boot))
json.dump(res, open(os.path.join(OUT, "palcp_verification.json"), "w", encoding="utf-8"),
          indent=1, default=float)
print("\nsaved palcp_verification.json")
