"""Transfer / reliability experiments: leave-one-model-out, cross-model matrix,
leave-one-subfield-out, and cross-intensity transfer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import exp_core as ec

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
df = corpus.load().reset_index(drop=True)
S = ec.prepare_stylo(df)

MODELS = ["deepseek", "doubao", "kimi", "wenxin"]
SUBS = ["criminal", "civil_commercial", "administrative", "constitutional",
        "legal_theory_history", "other"]
split = df.group_split.values
prov = df.provider.values
sub = df.subfield.values
is_orig = (df.operation == "original").values
ed = ~is_orig


def fit_eval(itr, ite, ycol, kind="hyb"):
    ytr, yte = df[ycol].values[itr], df[ycol].values[ite]
    FM = S
    det = ec.Detector(kind, C=1.0, balanced=(ycol == "intervention_present"))
    det.fit(df.abstract_clean.values[itr].tolist(), ytr, FM[itr])
    p1, P = det.predict_proba(df.abstract_clean.values[ite].tolist(), FM[ite])
    return yte, p1, P, det


rows = []

# Match the in-domain comparator to the transfer model and training protocol.
baseline = []
for task, ycol in [("presence", "intervention_present"), ("type", "operation"),
                   ("level", "level")]:
    mask = ed if task == "type" else np.ones(len(df), bool)
    itr = np.where((split == "train") & mask)[0]
    ite = np.where((split == "test") & mask)[0]
    yte, p1, P, det = fit_eval(itr, ite, ycol)
    if p1 is not None:
        r = ec.binary_report(yte, p1)
    else:
        pred = det.clf.classes_[P.argmax(1)]
        r = ec.multi_report(yte, pred, P)
        if task == "level":
            r.update(ec.ordinal_report(yte, pred))
    r.update(task=task, detector="hyb", C=1.0)
    baseline.append(r)
pd.DataFrame(baseline).to_csv(OUT / "transfer_baselines.csv", index=False, encoding="utf-8-sig")

# ---------- 1. leave-one-model-out ----------
print("== LOMO ==", flush=True)
for m in MODELS:
    for task, ycol in [("presence", "intervention_present"), ("type", "operation"),
                       ("level", "level")]:
        if task == "type":
            itr = np.where((split == "train") & (prov != m) & ed)[0]
            ite = np.where((split == "test") & (prov == m))[0]
        else:
            itr = np.where((split == "train") & (prov != m))[0]
            ite = np.where((split == "test") & ((prov == m) | is_orig))[0]
        yte, p1, P, det = fit_eval(itr, ite, ycol)
        if p1 is not None:
            r = ec.binary_report(yte, p1)
        else:
            pred = det.clf.classes_[P.argmax(1)]
            r = ec.multi_report(yte, pred, P)
            if task == "level":
                r.update(ec.ordinal_report(yte, pred))
        r.update(protocol="LOMO", heldout=m, task=task)
        rows.append(r)
        print(m, task, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

# ---------- 2. cross-model presence matrix ----------
print("\n== cross-model presence matrix ==", flush=True)
mat = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
mat5 = mat.copy()
for m1 in MODELS:
    itr = np.where((split == "train") & (is_orig | (prov == m1)))[0]
    ytr = df.intervention_present.values[itr]
    det = ec.Detector("hyb", balanced=True).fit(df.abstract_clean.values[itr].tolist(), ytr, S[itr])
    for m2 in MODELS:
        ite = np.where((split == "test") & (is_orig | (prov == m2)))[0]
        yte = df.intervention_present.values[ite]
        p1, _ = det.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
        mat.loc[m1, m2] = ec.roc_auc_score(yte, p1)
        mat5.loc[m1, m2] = ec.tpr_at_fpr(yte, p1, 0.05)
mat.to_csv(os.path.join(OUT, "cross_model_auc.csv"), encoding="utf-8-sig")
mat5.to_csv(os.path.join(OUT, "cross_model_tpr5.csv"), encoding="utf-8-sig")
print(mat.round(3).to_string(), flush=True)

# ---------- 3. leave-one-subfield-out ----------
print("\n== LOSO (subfield) ==", flush=True)
for s in SUBS:
    for task, ycol in [("presence", "intervention_present"), ("type", "operation"), ("level", "level")]:
        m_tr = np.ones(len(df), bool) if task != "type" else ed
        m_te = np.ones(len(df), bool) if task != "type" else ed
        itr = np.where((split == "train") & (sub != s) & m_tr)[0]
        ite = np.where((split == "test") & (sub == s) & m_te)[0]
        if len(ite) < 30 or len(np.unique(df[ycol].values[ite])) < 2:
            continue
        yte, p1, P, det = fit_eval(itr, ite, ycol)
        if p1 is not None:
            r = ec.binary_report(yte, p1)
        else:
            pred = det.clf.classes_[P.argmax(1)]
            r = ec.multi_report(yte, pred, P)
            if task == "level":
                r.update(ec.ordinal_report(yte, pred))
        r.update(protocol="LOSO", heldout=s, task=task)
        rows.append(r)
        print(s, task, round(r.get("auc", r.get("mf1", 0)), 3), flush=True)

# ---------- 4. cross-intensity transfer ----------
print("\n== cross-intensity ==", flush=True)
light = ["grammar_correction", "academic_polish"]
heavy = ["sentence_restructure", "abstract_rewrite"]
for src_name, src_ops, tgt_name, tgt_ops in [
        ("light", light, "heavy", heavy), ("heavy", heavy, "light", light),
        ("E1", ["grammar_correction"], "E4", ["abstract_rewrite"]),
        ("E2", ["academic_polish"], "E3", ["sentence_restructure"])]:
    itr = np.where((split == "train") & (is_orig | df.operation.isin(src_ops).values))[0]
    ite = np.where((split == "test") & (is_orig | df.operation.isin(tgt_ops).values))[0]
    ytr = df.intervention_present.values[itr]
    det = ec.Detector("hyb", balanced=True).fit(df.abstract_clean.values[itr].tolist(), ytr, S[itr])
    yte = df.intervention_present.values[ite]
    p1, _ = det.predict_proba(df.abstract_clean.values[ite].tolist(), S[ite])
    r = ec.binary_report(yte, p1)
    r.update(protocol="crossintensity", train_on=src_name, test_on=tgt_name, task="presence")
    rows.append(r)
    print(src_name, "->", tgt_name, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_results.csv"), index=False, encoding="utf-8-sig")
print("\nsaved", len(rows), "rows")
