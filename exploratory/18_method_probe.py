"""Method probe: compare candidate contributions on identical features.

(A) Joint provenance lattice: one 17-way model over {H0} x {E1..E4} x {4 models},
    marginalised to presence / level / model / type.
(B) Hierarchical two-head model: P(present) x P(level | present).
(C) Selective prediction: reject by predicted edit depth vs by confidence/entropy.
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
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import feats
from provenance.exp_core import hierarchical_level_probabilities

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
split = df.group_split.values
itr = np.where(split == "train")[0]
iva = np.where(split == "dev")[0]
ite = np.where(split == "test")[0]

OPS = ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]
MODS = ["deepseek", "doubao", "kimi", "wenxin"]

vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=3,
                      sublinear_tf=True, max_features=300000)
Xtr = vec.fit_transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[itr]])
Xva = vec.transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[iva]])
Xte = vec.transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[ite]])

y_pres = df.intervention_present.values
y_op = df.operation.values
y_mod = df.provider.values
y_lvl = df.level.values
is_ed = y_pres == 1

prov = np.array([f"{p}|{o}" for p, o in zip(y_mod, y_op)])
prov[~is_ed] = "H0"
print("joint classes in train:", len(np.unique(prov[itr])))

R = {}

# ---------------- (A) joint 17-way ----------------
levels = ["H0"] + [f"{m}|{o}" for m in MODS for o in OPS]
h0i = levels.index("H0")

best = (-9, None, None)
for C in [0.25, 1.0, 4.0]:
    m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs").fit(Xtr, prov[itr])
    P = m.predict_proba(Xva)
    h0 = list(m.classes_).index("H0")
    s = roc_auc_score(y_pres[iva], 1 - P[:, h0])
    if s > best[0]:
        best = (s, C, m)
_, C17, m17 = best
cls17 = list(m17.classes_)
P17 = m17.predict_proba(Xte)
R["joint17_C"] = C17
R["joint17_presence_auc"] = float(roc_auc_score(y_pres[ite], 1 - P17[:, cls17.index("H0")]))

Plev = np.zeros((len(ite), 5))
Pops = np.zeros((len(ite), 4))
Pmod = np.zeros((len(ite), 4))
for j, c in enumerate(cls17):
    if c == "H0":
        Plev[:, 0] += P17[:, j]
    else:
        m_, o_ = c.split("|")
        Plev[:, 1 + OPS.index(o_)] += P17[:, j]
        Pops[:, OPS.index(o_)] += P17[:, j]
        Pmod[:, MODS.index(m_)] += P17[:, j]
me = is_ed[ite]
R["joint17_level_qwk"] = float(cohen_kappa_score(y_lvl[ite], Plev.argmax(1), weights="quadratic"))
R["joint17_type_mf1"] = float(f1_score(y_op[ite][me], np.array(OPS)[Pops.argmax(1)][me], average="macro"))
R["joint17_model_mf1"] = float(f1_score(y_mod[ite][me], np.array(MODS)[Pmod.argmax(1)][me], average="macro"))
print("A joint-17:", {k: round(v, 3) for k, v in R.items() if isinstance(v, float)})

# ---------------- (B) hierarchical ----------------
best = (-9, None, None)
for C in [0.25, 1.0, 4.0]:
    g = LogisticRegression(C=C, max_iter=3000, solver="liblinear", class_weight="balanced",
                           random_state=0).fit(Xtr, y_pres[itr])
    s = roc_auc_score(y_pres[iva], g.predict_proba(Xva)[:, 1])
    if s > best[0]:
        best = (s, C, g)
_, Cg, gap = best
p_gate = gap.predict_proba(Xte)[:, 1]
R["hier_C"] = Cg
R["hier_presence_auc"] = float(roc_auc_score(y_pres[ite], p_gate))

tr_e = itr[is_ed[itr]]
te_e = ite[is_ed[ite]]
gl = LogisticRegression(C=0.25, max_iter=3000, solver="lbfgs")
gl.fit(vec.transform([feats.clean_for_ngram(t) for t in df.abstract_clean.values[tr_e]]), y_op[tr_e])
P4 = gl.predict_proba(Xte)
Plev_h = hierarchical_level_probabilities(
    p_gate, P4[:, [list(gl.classes_).index(j) for j in OPS]])
R["hier_level_qwk"] = float(cohen_kappa_score(y_lvl[ite], Plev_h.argmax(1), weights="quadratic"))
R["hier_type_mf1"] = float(f1_score(y_op[te_e], np.array(gl.classes_)[P4[is_ed[ite]].argmax(1)], average="macro"))
print("B hierarchical:", {k: round(v, 3) for k, v in R.items() if isinstance(v, float)})

# ---------------- (C) selective prediction ----------------
reg = Ridge(alpha=1.0, solver="lsqr").fit(Xtr, df.normalized_edit_distance.values[itr])
ghat = reg.predict(Xte)
g_true = df.normalized_edit_distance.values[ite]

P5 = Plev_h
conf = P5.max(1)
ent = -(P5 * np.log(P5 + 1e-12)).sum(1)
pred5 = np.array(["original"] + OPS)[P5.argmax(1)]
correct = (pred5 == y_op[ite]).astype(int)


def acc_at_coverage(score, correct, tgt):
    idx = np.argsort(score)
    k = max(1, int(round(tgt * len(idx))))
    return float(correct[idx[-k:]].mean())


R["g_spearman"] = float(pd.Series(ghat).corr(pd.Series(g_true), method="spearman"))
R["g_corr_conf"] = float(np.corrcoef(ghat, conf)[0, 1])
print(f"\nSpearman(g_hat, true)={R['g_spearman']:.3f}  corr(g_hat, confidence)={R['g_corr_conf']:.3f}")

print("\nselective accuracy at coverage 50/70/90% (higher score = kept first):")
for name, sc in [("confidence", conf), ("neg_entropy", -ent), ("pred_magnitude", ghat),
                 ("true_magnitude_oracle", g_true)]:
    vals = [acc_at_coverage(sc, correct, t) for t in (0.5, 0.7, 0.9)]
    for t, v in zip((50, 70, 90), vals):
        R[f"selacc_{name}_{t}"] = v
    print(f"  {name:24s}", [round(v, 3) for v in vals])

# magnitude gate: abstain when predicted depth is below tau
print("\nmagnitude gate (abstain below tau):")
for tau in [0.02, 0.05, 0.10]:
    rej = ghat < tau
    ret = ~rej
    R[f"gate_{tau}_reject"] = float(rej.mean())
    R[f"gate_{tau}_acc_retained"] = float(correct[ret].mean()) if ret.sum() else float("nan")
    human = (y_pres[ite] == 0)
    if (ret & human).any():
        R[f"gate_{tau}_human_recall_retained"] = float((pred5[ret & human] == "original").mean())
    print(f"  tau={tau:.2f}: reject {rej.mean():.2f}  acc|retained {correct[ret].mean():.3f}"
          f"  recall on retained humans {R.get(f'gate_{tau}_human_recall_retained', float('nan')):.3f}")

json.dump(R, open(os.path.join(OUT, "method_probe.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\nsaved method_probe.json")
