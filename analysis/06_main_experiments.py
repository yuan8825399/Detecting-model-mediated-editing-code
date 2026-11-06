"""Main detection experiments on the official group-disjoint split (dev-tuned)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, json, os, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import exp_core as ec
from sklearn.metrics import roc_auc_score

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)

df = corpus.load().reset_index(drop=True)
S = ec.prepare_stylo(df)
EMB_PATH = PROCESSED_DIR / "emb_mean.npy"
emb = np.load(EMB_PATH) if os.path.exists(EMB_PATH) else None
print("stylo", S.shape, "emb", None if emb is None else emb.shape, flush=True)

KINDS = os.environ.get("KINDS", "sty,ngr,hyb,bert").split(",")
if "bert" in KINDS and emb is None:
    raise FileNotFoundError("BERT requested but emb_mean.npy is missing; run embeddings first")
if emb is not None and emb.shape != (len(df), 768):
    raise ValueError("embedding matrix does not match the corpus rows and dimensions")
print("running kinds:", KINDS, flush=True)

split = df.group_split.values
GRID = [0.25, 1.0, 4.0, 16.0]


def idxs(mask, which):
    return np.where(np.asarray(mask, bool) & (split == which))[0]


def run_task(name, mask, ycol, kind, binary, nboot=300):
    tag = "bert" if kind == "bert" else "base"
    itr, iva, ite = idxs(mask, "train"), idxs(mask, "dev"), idxs(mask, "test")
    FM = emb if kind == "bert" else S
    best, bestC = -1, 1.0
    for C in GRID:
        det = ec.Detector(kind, C=C, balanced=binary)
        det.fit(df.abstract_clean.values[itr].tolist(), df[ycol].values[itr], FM[itr])
        if binary:
            _, P = det.predict_proba(df.abstract_clean.values[iva].tolist(), FM[iva])
            sc = roc_auc_score(df[ycol].values[iva], P[:, 1])
        else:
            pr = det.clf.classes_[det.predict_proba(df.abstract_clean.values[iva].tolist(), FM[iva])[1].argmax(1)]
            sc = (pr == df[ycol].values[iva]).mean()
        if sc > best:
            best, bestC = sc, C
    det = ec.Detector(kind, C=bestC, balanced=binary)
    t0 = time.time()
    det.fit(df.abstract_clean.values[itr].tolist(), df[ycol].values[itr], FM[itr])
    p1, P = det.predict_proba(df.abstract_clean.values[ite].tolist(), FM[ite])
    fit_s = time.time() - t0
    yte = df[ycol].values[ite]
    g = df.source_id.values[ite]
    if binary:
        rep = ec.binary_report(yte, p1)
        rep["auc_ci"] = ec.group_bootstrap(yte, p1, lambda a, b: roc_auc_score(a, b), g, nboot)
        rep["tpr5_ci"] = ec.group_bootstrap(yte, p1, lambda a, b: ec.tpr_at_fpr(a, b, 0.05), g, nboot)
        np.save(os.path.join(OUT, f"pred_{name}_{tag}_{kind}.npy"), np.vstack([ite, p1]).T)
    else:
        pred = det.clf.classes_[P.argmax(1)]
        rep = ec.multi_report(yte, pred, P)
        if name == "level":
            rep.update(ec.ordinal_report(yte, pred))
        rep["mf1_ci"] = ec.group_bootstrap(yte, pred, lambda a, b: ec.f1_score(a, b, average="macro"), g, nboot)
        np.save(os.path.join(OUT, f"pred_{name}_{tag}_{kind}.npy"), np.vstack([ite, pred]).T)
    rep.update(task=name, detector=kind, n_train=len(itr), n_test=len(ite),
               dev_score=best, C=bestC, fit_sec=round(fit_s, 1))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}, flush=True)
    return rep


rows = []
tasks = [
    ("presence", np.ones(len(df), bool), "intervention_present", True),
    ("presence_ne", corpus.effective_presence_mask(df),
     "intervention_present", True),
    ("type", (df.intervention_present == 1).values, "operation", False),
    ("type_ne", ((df.intervention_present == 1) & (df.identical == 0)).values, "operation", False),
    ("level", np.ones(len(df), bool), "level", False),
    ("model", (df.intervention_present == 1).values, "provider", False),
]
for name, mask, ycol, binary in tasks:
    for kind in KINDS:
        rows.append(run_task(name, mask, ycol, kind, binary))

results = pd.DataFrame(rows)
for tag, mask in [("base", results.detector != "bert"), ("bert", results.detector == "bert")]:
    if mask.any():
        results[mask].to_csv(OUT / f"main_results_{tag}.csv", index=False, encoding="utf-8-sig")
print("\n=== summary ===")
print(pd.DataFrame(rows)[["task", "detector", "C", "auc", "tpr_at_5pct", "ece", "mf1", "qwk", "acc"]]
      .round(3).to_string())
