"""Robustness analyses: per-type detectability, dose-response, calibration,
paired discrimination, human-FPR stability, cross-era transfer, ablations, ceiling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import exp_core as ec

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
df = corpus.load().reset_index(drop=True)
S = ec.prepare_stylo(df)
split = df.group_split.values
itr = np.where(split == "train")[0]
ite = np.where(split == "test")[0]

arr = np.load(os.path.join(OUT, "pred_presence_base_ngr.npy"))
assert (arr[:, 0].astype(int) == ite).all()
p = arr[:, 1]
te = df.iloc[ite].reset_index(drop=True)
te["p"] = p
orig_te = te[te.operation == "original"]
icb = np.where(split == "calibration")[0]
threshold_detector = ec.Detector("ngr", C=4.0, balanced=True).fit(
    df.abstract_clean.values[itr].tolist(), df.intervention_present.values[itr], S[itr])
p_cb, _ = threshold_detector.predict_proba(df.abstract_clean.values[icb].tolist(), S[icb])
h_cb = p_cb[df.intervention_present.values[icb] == 0]
thr5 = np.quantile(h_cb, 0.95)
thr1 = np.quantile(h_cb, 0.99)
print(f"calibration thresholds: nominal 5%={thr5:.3f}, nominal 1%={thr1:.3f}")

# ---------- 1. detectability by edit type (per-operation AUC) ----------
rows = []
for opn in ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]:
    sub = te[(te.operation == "original") | (te.operation == opn)]
    y = sub.intervention_present.values
    rows.append(dict(operation=opn, n_edit=int((y == 1).sum()),
                     auc=roc_auc_score(y, sub.p.values),
                     tpr5=float((sub[sub.operation == opn].p >= thr5).mean()),
                     tpr1=float((sub[sub.operation == opn].p >= thr1).mean()),
                     mean_p=float(sub[sub.operation == opn].p.mean())))
per_op = pd.DataFrame(rows)
print("\n=== detectability by edit type (vs human originals) ===")
print(per_op.round(3).to_string(index=False))
per_op.to_csv(os.path.join(OUT, "per_type_detectability.csv"), index=False, encoding="utf-8-sig")

# per type x model
rows = []
for opn in ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]:
    for m in ["deepseek", "doubao", "kimi", "wenxin"]:
        e = te[(te.operation == opn) & (te.provider == m)]
        sub = pd.concat([orig_te, e])
        y = sub.intervention_present.values
        rows.append(dict(operation=opn, provider=m, n=len(e),
                         auc=roc_auc_score(y, sub.p.values),
                         tpr5=float((e.p >= thr5).mean()),
                         mean_ed=float(e.normalized_edit_distance.mean())))
PT = pd.DataFrame(rows)
PT.to_csv(os.path.join(OUT, "per_type_model_detectability.csv"), index=False, encoding="utf-8-sig")
print("\n=== TPR@5%FPR by type x model ===")
print(PT.pivot_table(index="operation", columns="provider", values="tpr5").round(3))
print("\n=== mean edit distance by type x model ===")
print(PT.pivot_table(index="operation", columns="provider", values="mean_ed").round(3))

# ---------- 2. dose-response ----------
ed = te[te.operation != "original"].copy()
ed["flagged5"] = (ed.p >= thr5).astype(int)
ed["flagged1"] = (ed.p >= thr1).astype(int)
bins = [-1e-9, 1e-9, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 10]
labels = ["0", "(0,.01]", "(.01,.05]", "(.05,.10]", "(.10,.20]",
          "(.20,.30]", "(.30,.40]", ">.40"]
ed["mag_bin"] = pd.cut(ed.normalized_edit_distance, bins=bins, labels=labels, right=True)
dose = ed.groupby("mag_bin", observed=True).agg(
    n=("p", "size"), mean_ed=("normalized_edit_distance", "mean"),
    mean_p=("p", "mean"), tpr5=("flagged5", "mean"), tpr1=("flagged1", "mean")).reset_index()
print("\n=== dose-response ===")
print(dose.round(3).to_string(index=False))
dose.to_csv(os.path.join(OUT, "dose_response.csv"), index=False, encoding="utf-8-sig")

# correlation between per-document edit magnitude and score
from numpy import corrcoef
r = np.corrcoef(ed.normalized_edit_distance, ed.p)[0, 1]
print("corr(edit magnitude, detector score) =", round(float(r), 3), " n=", len(ed))

# ---------- 3. calibration (unweighted model) ----------
det = ec.Detector("ngr", C=4.0, balanced=False)
det.fit(df.abstract_clean.values[itr].tolist(), df.intervention_present.values[itr], S[itr])
pu, _ = det.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
yu = df.intervention_present.values[ite]
cal_all = dict(ece=ec.ece(yu, pu), brier=float(np.mean((pu - yu) ** 2)),
               auc=roc_auc_score(yu, pu), mean_p=float(pu.mean()), obs=float(yu.mean()))
print("\n=== calibration (unweighted, natural prevalence) ===", {k: round(v, 4) for k, v in cal_all.items()})
rel = []
edges = np.linspace(0, 1, 11)
for i in range(10):
    m = (pu >= edges[i]) & (pu <= edges[i + 1] if i == 9 else pu < edges[i + 1])
    if m.sum() == 0:
        continue
    rel.append(dict(bin=f"{edges[i]:.1f}-{edges[i+1]:.1f}", n=int(m.sum()),
                    mean_p=float(pu[m].mean()), obs=float(yu[m].mean())))
REL = pd.DataFrame(rel)
REL.to_csv(os.path.join(OUT, "reliability.csv"), index=False, encoding="utf-8-sig")
print(REL.round(3).to_string(index=False))

# calibration by operation stratum
cal = []
for opn in ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]:
    m = (te.operation == opn).values
    cal.append(dict(operation=opn, mean_p=float(pu[m].mean()), ece=ec.ece(np.ones(m.sum()), pu[m])))
CAL = pd.DataFrame(cal)
print("\n=== mean LLM-probability by edit type (unweighted model) ===")
print(CAL.round(3).to_string(index=False))
CAL.to_csv(os.path.join(OUT, "calibration_by_type.csv"), index=False, encoding="utf-8-sig")

# ---------- 4. paired discrimination ----------
o = orig_te.set_index("source_id").p
pair = []
for (sid, opn), g in ed.groupby(["source_id", "operation"]):
    if sid not in o.index:
        continue
    for _, r in g.iterrows():
        pair.append((sid, opn, r.provider, r.subfield, r.era,
                     float(o[sid]), float(r.p), float(r.p - o[sid])))
PR = pd.DataFrame(pair, columns=["source_id", "operation", "provider", "subfield",
                                 "era", "p_orig", "p_edit", "delta"])
PR["win"] = (PR.delta > 0).astype(int)
print("\n=== paired discrimination ===")
print(PR.pivot_table(index="operation", columns="provider", values="win").round(3))
print("overall:", round(PR.win.mean(), 3))
PR.to_csv(os.path.join(OUT, "paired_discrimination.csv"), index=False, encoding="utf-8-sig")

# ---------- 5. human-FPR stability ----------
print("\n=== FPR on human text by subfield / era ===")
fpr_s = orig_te.groupby("subfield").apply(
    lambda d: pd.Series(dict(n=len(d), mean_p=d.p.mean(), fpr5=float((d.p >= thr5).mean()))),
    include_groups=False)
fpr_e = orig_te.groupby("era").apply(
    lambda d: pd.Series(dict(n=len(d), mean_p=d.p.mean(), fpr5=float((d.p >= thr5).mean()))),
    include_groups=False)
print(fpr_s.round(3).to_string()); print(fpr_e.round(3).to_string())
fpr_s.to_csv(os.path.join(OUT, "fpr_subfield.csv"), encoding="utf-8-sig")
fpr_e.to_csv(os.path.join(OUT, "fpr_era.csv"), encoding="utf-8-sig")

# ---------- 6. cross-era transfer ----------
print("\n=== cross-era transfer ===")
eras = ["2000-2004", "2005-2009", "2010-2014", "2015-2020"]
M = pd.DataFrame(index=eras, columns=eras, dtype=float)
for e1 in eras:
    i_tr = np.where((split != "test") & (df.era == e1).values)[0]
    if len(np.unique(df.intervention_present.values[i_tr])) < 2:
        continue
    d = ec.Detector("ngr", C=4.0, balanced=True).fit(
        df.abstract_clean.values[i_tr].tolist(), df.intervention_present.values[i_tr], S[i_tr])
    for e2 in eras:
        i_te = np.where((split == "test") & (df.era == e2).values)[0]
        pb, _ = d.predict_proba(df.abstract_clean.values[i_te].tolist(), S[i_te])
        M.loc[e1, e2] = roc_auc_score(df.intervention_present.values[i_te], pb)
print(M.round(3).to_string())
M.to_csv(os.path.join(OUT, "cross_era_auc.csv"), encoding="utf-8-sig")

# ---------- 7. ablations ----------
print("\n== ablations ==", flush=True)
ytr, yte = df.intervention_present.values[itr], df.intervention_present.values[ite]
LEN = ["log_chars", "n_sent", "mean_slen", "sd_slen", "cv_slen", "max_slen",
       "len_per_clause", "commas_per_sent"]
rows = []
for name, kw in [("hyb_full", dict(kind="hyb")), ("hyb_no_length", dict(kind="hyb", drop=LEN)),
                 ("sty_only", dict(kind="sty")), ("sty_no_length", dict(kind="sty", drop=LEN)),
                 ("ngram_only", dict(kind="ngr"))]:
    d = ec.Detector(C=4.0, **kw).fit(df.abstract_clean.values[itr].tolist(), ytr, S[itr])
    pb, _ = d.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
    r = ec.binary_report(yte, pb); r["variant"] = name; rows.append(r)
    print(name, round(r["auc"], 4), round(r["tpr_at_5pct"], 3), flush=True)

subs = pd.get_dummies(df.subfield).values.astype(float)
meta = np.hstack([df[["char_count"]].values / 100.0, subs])
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(meta[itr], ytr)
pm = clf.predict_proba(meta[ite])[:, 1]
r = ec.binary_report(yte, pm); r["variant"] = "metadata_only_control"; rows.append(r)
print("metadata_only", round(r["auc"], 4), round(r["tpr_at_5pct"], 3))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablations.csv"), index=False, encoding="utf-8-sig")

# ---------- 8. ceiling ----------
ceil = df[df.operation != "original"].groupby("operation").agg(
    identical_rate=("identical", "mean"), n=("identical", "size"),
    mean_ed=("normalized_edit_distance", "mean"),
    med_ed=("normalized_edit_distance", "median")).reset_index()
ceil["max_recall"] = 1 - ceil.identical_rate
print("\n=== ceiling ===")
print(ceil.round(3).to_string(index=False))
ceil.to_csv(os.path.join(OUT, "ceiling.csv"), index=False, encoding="utf-8-sig")
print("\ndone")
