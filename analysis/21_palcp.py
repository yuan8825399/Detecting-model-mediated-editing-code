"""PAL-CP final evaluation: one joint lattice model, two decision rules compared
on identical scores.

Rule A (conventional): threshold fixed on the calibration partition at 5% FPR.
Rule B (PAL-CP): class-conditional split conformal -> set-valued verdict with an
                  explicit undecidable region.

Both are scored by the same criterion: how many *silent* errors does the rule issue
per 1,000 submissions, where a silent error is a wrong verdict reported without any
uncertainty marker.
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
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import feats
from provenance.conformal import quantile_nonconformity
from provenance.exp_core import group_bootstrap

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

y = df.intervention_present.values
prov = np.array([f"{m}|{o}" for m, o in zip(df.provider.values, df.operation.values)])
prov[y == 0] = "H0"

clf = LogisticRegression(C=4.0, max_iter=3000, solver="lbfgs").fit(Xtr, prov[itr])
cls = list(clf.classes_)
h0 = cls.index("H0")
pres_va = 1 - clf.predict_proba(Xva)[:, h0]
pres_te = 1 - clf.predict_proba(Xte)[:, h0]
yte = y[ite]
opte = df.operation.values[ite]
print("presence AUC:", round(roc_auc_score(yte, pres_te), 4))

alpha = 0.10
# ---- Rule A: threshold at 5% FPR on the calibration partition ----
thrA = float(np.quantile(pres_va[y[iva] == 0], 0.95))
flagA = pres_te >= thrA
print(f"\nRule A threshold (5% FPR on calibration) = {thrA:.4f}; "
      f"test FPR = {flagA[yte == 0].mean():.3f}, recall = {flagA[yte == 1].mean():.3f}")

# ---- Rule B: class-conditional conformal ----
s_va = np.where(y[iva] == 1, 1 - pres_va, pres_va)
def qh(s, a):
    return quantile_nonconformity(s, a)
q_pos = qh(s_va[y[iva] == 1], alpha)
q_neg = qh(s_va[y[iva] == 0], alpha)
inc_p = (1 - pres_te) <= q_pos + 1e-12
inc_a = pres_te <= q_neg + 1e-12
dec = inc_p ^ inc_a
undec = inc_p & inc_a
empty = (~inc_p) & (~inc_a)

verdict = np.where(inc_p, 1, 0)

def summary(name, issued_mask, verdict_vals):
    issued = issued_mask.sum()
    err = ((verdict_vals == 1) != (yte == 1)) & issued_mask
    return dict(rule=name, issued=int(issued), issued_rate=float(issued_mask.mean()),
                silent_errors=int(err.sum()),
                silent_error_per_1000=float(1000 * err.sum() / len(yte)),
                abstained=int((~issued_mask).sum()),
                precision_present=float(yte[issued_mask & (verdict_vals == 1)].mean())
                if (issued_mask & (verdict_vals == 1)).any() else None,
                precision_absent=float(1 - yte[issued_mask & (verdict_vals == 0)].mean())
                if (issued_mask & (verdict_vals == 0)).any() else None)

A = summary("Threshold (5% FPR, calibration)", np.ones(len(yte), bool), flagA.astype(int))
B = summary("PAL-CP (class-conditional conformal)", dec, verdict)
print("\n=== decision rules on identical scores ===")
for r in (A, B):
    print(f"{r['rule']:42s} issued={r['issued']:5d} ({r['issued_rate']*100:5.1f}%)  "
          f"silent errors={r['silent_errors']:4d} ({r['silent_error_per_1000']:6.1f}/1000)  "
          f"abstained={r['abstained']:4d}")
print(f"\nPAL-CP: precision(present)={B['precision_present']:.3f}, "
      f"precision(absent)={B['precision_absent']:.3f}, undecidable={int(undec.sum())}, empty={int(empty.sum())}")

# ---- coverage diagnostics ----
cov = np.where(yte == 1, inc_p, inc_a)
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0., (c - h) / d), min(1., (c + h) / d))
cov_ci = group_bootstrap(yte, cov, lambda y, c: c.mean(),
                         df.source_id.values[ite], n_boot=2000, seed=7)
print(f"\ncoverage overall {cov.mean():.3f} [{cov_ci[0]:.3f}, {cov_ci[1]:.3f}]; "
      f"edited {cov[yte==1].mean():.3f}; human {cov[yte==0].mean():.3f}")

# ---- where do the false-absent verdicts come from ----
fa = dec & (verdict == 0) & (yte == 1)
comp = pd.Series(opte[fa]).value_counts()
print("\nfalse-absent verdicts by true operation:")
print(comp.to_string())
e1_share = float((opte[fa] == "grammar_correction").mean()) if fa.sum() else float("nan")
print("share E1:", round(e1_share, 3))

# ---- per-condition behaviour of the rule ----
rows = []
g = df.normalized_edit_distance.values[ite]
for op_label, msk in ([("human (original)", yte == 0)] +
                      [(o, opte == o) for o in OPS]):
    rows.append(dict(condition=op_label, n=int(msk.sum()),
                     decisive=float(dec[msk].mean()),
                     inc_present=float(inc_p[msk].mean()),
                     inc_absent=float(inc_a[msk].mean()),
                     coverage=float(cov[msk].mean())))
C = pd.DataFrame(rows)
print("\nper-condition behaviour of PAL-CP:")
print(C.round(3).to_string(index=False))
C.to_csv(os.path.join(OUT, "palcp_conditions.csv"), index=False, encoding="utf-8-sig")

# ---- decisiveness vs magnitude ----
bins = [-1e-9, 1e-9, 0.05, 0.15, 0.30, 10]
labs = ["0", "(0,0.05]", "(0.05,0.15]", "(0.15,0.30]", ">0.30"]
bb = np.asarray(pd.cut(g, bins, labels=labs).astype(str))
rows = []
for lab in labs:
    msk = bb == lab
    if not msk.sum():
        continue
    rows.append(dict(bin=lab, n=int(msk.sum()), decisive=float(dec[msk].mean()),
                     coverage=float(cov[msk].mean()),
                     abstain=float(undec[msk].mean())))
D = pd.DataFrame(rows)
print("\ndecisiveness vs edit magnitude:")
print(D.round(3).to_string(index=False))
D.to_csv(os.path.join(OUT, "palcp_decisiveness.csv"), index=False, encoding="utf-8-sig")

# ---- joint model vs task-specific baselines ----
R = json.load(open(os.path.join(OUT, "method_probe.json"), encoding="utf-8"))
main_results = pd.read_csv(OUT / "main_results_base.csv")
def baseline(task, metric):
    return float(main_results[(main_results.task == task) &
                              (main_results.detector == "ngr")][metric].iloc[0])
T = pd.DataFrame([
    dict(task="Presence (AUC)", baseline=baseline("presence", "auc"), joint=R["joint17_presence_auc"],
         hier=R["hier_presence_auc"]),
    dict(task="Type (macro-F1)", baseline=baseline("type", "mf1"), joint=R["joint17_type_mf1"],
         hier=R["hier_type_mf1"]),
    dict(task="Level (QWK)", baseline=baseline("level", "qwk"), joint=R["joint17_level_qwk"],
         hier=R["hier_level_qwk"]),
    dict(task="Model (macro-F1)", baseline=baseline("model", "mf1"), joint=R["joint17_model_mf1"], hier=np.nan),
])
print("\n=== single-lattice model vs task-specific baselines ===")
print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(OUT, "lattice_vs_baselines.csv"), index=False, encoding="utf-8-sig")

# ---- Rule C (diagnostic): marginal / global conformal, same scores ----
q_glob = qh(s_va, alpha)
inc_p_g = (1 - pres_te) <= q_glob + 1e-12
inc_a_g = pres_te <= q_glob + 1e-12
cov_g = np.where(yte == 1, inc_p_g, inc_a_g)
print("\n=== diagnostic: marginal (non-class-conditional) conformal ===")
print(f"q_glob={q_glob:.4f}  coverage overall={cov_g.mean():.3f}  "
      f"edited={cov_g[yte==1].mean():.3f}  human={cov_g[yte==0].mean():.3f}")
print(f"human class included in set: {inc_p_g[yte==0].mean():.3f}, "
      f"absent included: {inc_a_g[yte==0].mean():.3f}")

out = dict(alpha=alpha, thrA=thrA, ruleA=A, ruleB=B,
           coverage=[float(cov_ci[0]), float(cov.mean()), float(cov_ci[1])],
           coverage_edited=float(cov[yte == 1].mean()),
           coverage_human=float(cov[yte == 0].mean()),
           q_pos=q_pos, q_neg=q_neg, undecidable=int(undec.sum()), empty=int(empty.sum()),
           false_absent=int(fa.sum()), e1_share_of_false_absent=e1_share,
           presence_auc=float(roc_auc_score(yte, pres_te)),
           marginal_conformal=dict(q=float(q_glob), coverage=float(cov_g.mean()),
                                   cov_edited=float(cov_g[yte == 1].mean()),
                                   cov_human=float(cov_g[yte == 0].mean()),
                                   empty=int((~inc_p_g & ~inc_a_g).sum()),
                                   empty_human=int((~inc_p_g & ~inc_a_g & (yte == 0)).sum()),
                                   present_human=int((inc_p_g & ~inc_a_g & (yte == 0)).sum())))
json.dump(out, open(os.path.join(OUT, "palcp.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)

np.savez(os.path.join(OUT, "palcp_test.npz"), pres=pres_te, inc_p=inc_p, inc_a=inc_a,
         y=yte, op=opte, g=g, flagA=flagA)
print("\nsaved palcp.json + palcp_test.npz")
