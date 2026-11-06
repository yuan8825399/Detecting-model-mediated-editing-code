"""Final candidate: PAL-CP (Provenance Audit Lattice with Conformal Prediction).

One 17-cell joint model -> marginal posteriors for the four audit questions ->
class-conditional (Mondrian) split-conformal sets with per-class coverage ->
decisiveness analysis tied to the detectability ceiling.
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

y_pres = df.intervention_present.values
prov = np.array([f"{m}|{o}" for m, o in zip(df.provider.values, df.operation.values)])
prov[df.intervention_present.values == 0] = "H0"

m = LogisticRegression(C=4.0, max_iter=3000, solver="lbfgs").fit(Xtr, prov[itr])
p_va = m.predict_proba(Xva)
p_te = m.predict_proba(Xte)
h0 = list(m.classes_).index("H0")
pres_va = 1 - p_va[:, h0]
pres_te = 1 - p_te[:, h0]
print("presence AUC on test:", round(roc_auc_score(y_pres[ite], pres_te), 4))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0., (c - h) / d), min(1., (c + h) / d))


def qhat(scores, alpha):
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(scores)[k - 1])


alpha = 0.10
# nonconformity for the binary presence question
s_va = np.where(y_pres[iva] == 1, 1 - pres_va, pres_va)
q_pos = qhat(s_va[y_pres[iva] == 1], alpha)   # threshold for "present" scores
q_neg = qhat(s_va[y_pres[iva] == 0], alpha)   # threshold for "absent" scores

# set: include "present" if 1 - p <= q_pos ; include "absent" if p <= q_neg
inc_present = (1 - pres_te) <= q_pos + 1e-12
inc_absent = pres_te <= q_neg + 1e-12
size = inc_present.astype(int) + inc_absent.astype(int)
covered = np.where(y_pres[ite] == 1, inc_present, inc_absent)
decisive = size == 1
print(f"\nq_pos={q_pos:.4f} (calibrated on {int((y_pres[iva]==1).sum())} edited), "
      f"q_neg={q_neg:.4f} (on {int((y_pres[iva]==0).sum())} human)")

dec_present = decisive & inc_present
dec_absent = decisive & inc_absent
print("\n=== presence, class-conditional conformal (alpha = 0.10) ===")
print(f"coverage overall      : {covered.mean():.3f}  [{wilson(int(covered.sum()), len(covered))[0]:.3f}, "
      f"{wilson(int(covered.sum()), len(covered))[1]:.3f}]")
for name, mask in [("on edited", y_pres[ite] == 1), ("on human", y_pres[ite] == 0)]:
    print(f"coverage {name:10s}: {covered[mask].mean():.3f}  (n={int(mask.sum())})")
print(f"decisive rate         : {decisive.mean():.3f}  (n={int(decisive.sum())})")
if dec_present.sum():
    print(f"  decisive 'present'  : {dec_present.sum()}  precision={y_pres[ite][dec_present].mean():.3f}")
if dec_absent.sum():
    print(f"  decisive 'absent'   : {dec_absent.sum()}  precision={1 - y_pres[ite][dec_absent].mean():.3f}")
print(f"  undecidable (both)  : {(size == 2).sum()}  empty set: {(size == 0).sum()}")

# decisiveness by true edit magnitude
g = df.normalized_edit_distance.values[ite]
bins = [-1e-9, 1e-9, 0.05, 0.15, 0.30, 10]
labs = ["0", "(0,0.05]", "(0.05,0.15]", "(0.15,0.30]", ">0.30"]
b = np.asarray(pd.cut(g, bins, labels=labs).astype(str))
print("\ndecisiveness and coverage by edit magnitude:")
rows = []
for lab in labs:
    msk = b == lab
    if not msk.sum():
        continue
    rows.append(dict(bin=lab, n=int(msk.sum()),
                     decisive=float(decisive[msk].mean()),
                     coverage=float(covered[msk].mean()),
                     p_decided_present=float(inc_present[msk].mean())))
    print(f"  {lab:12s} n={msk.sum():4d} decisive={decisive[msk].mean():.2f} "
          f"coverage={covered[msk].mean():.2f} incl_present={inc_present[msk].mean():.2f}")
D = pd.DataFrame(rows)

# comparison with the deployed fixed-threshold rule (from 14_recalib)
thr = json.load(open(os.path.join(OUT, "recalib.json"), encoding="utf-8"))["thr5"]
rule = pres_te >= thr
print("\n=== comparison: fixed-threshold rule at the calibration operating point ===")
print(f"flag rate={rule.mean():.3f}  precision={y_pres[ite][rule].mean():.3f}  "
      f"recall={rule[y_pres[ite]==1].mean():.3f}  fpr={rule[y_pres[ite]==0].mean():.3f}")
print("conformal decisive subset: "
      f"n={int(decisive.sum())} ({(decisive.mean()*100):.1f}% of corpus), "
      f"errors={int((~((dec_present & (y_pres[ite]==1)) | (dec_absent & (y_pres[ite]==0)))).sum())}")

R = dict(alpha=alpha, q_pos=q_pos, q_neg=q_neg,
         presence_auc=float(roc_auc_score(y_pres[ite], pres_te)),
         coverage_overall=float(covered.mean()),
         coverage_edited=float(covered[y_pres[ite] == 1].mean()),
         coverage_human=float(covered[y_pres[ite] == 0].mean()),
         decisive_rate=float(decisive.mean()),
         decisive_present=int(dec_present.sum()), decisive_absent=int(dec_absent.sum()),
         undecidable=int((size == 2).sum()), empty=int((size == 0).sum()),
         precision_present=float(y_pres[ite][dec_present].mean()) if dec_present.sum() else None,
         precision_absent=float(1 - y_pres[ite][dec_absent].mean()) if dec_absent.sum() else None,
         threshold_rule=dict(flag_rate=float(rule.mean()),
                             precision=float(y_pres[ite][rule].mean()),
                             recall=float(rule[y_pres[ite] == 1].mean()),
                             fpr=float(rule[y_pres[ite] == 0].mean())))
D.to_csv(os.path.join(OUT, "caps_decisiveness.csv"), index=False, encoding="utf-8-sig")
json.dump(R, open(os.path.join(OUT, "caps.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nsaved caps.json")
