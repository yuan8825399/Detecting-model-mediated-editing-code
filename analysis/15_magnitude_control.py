"""Does model attribution survive controlling for edit magnitude and length?
1. Attribution within edit-magnitude quintiles.
2. Magnitude-matched attribution (overlapping range only).
3. Attribution from length/punctuation features alone.
4. Document-level bootstrap CI for the attribution-vs-type difference.
5. Model x magnitude interaction description."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus
from provenance import exp_core as ec

OUT = RESULTS_DIR
df = corpus.load().reset_index(drop=True)
S = ec.prepare_stylo(df)
split = df.group_split.values
MODELS = ["deepseek", "doubao", "kimi", "wenxin"]
edited = (df.intervention_present == 1).values
R = {}

# ---------- 0. model x magnitude description ----------
e = df[edited]
tab = e.pivot_table(index="provider", columns="operation",
                    values="normalized_edit_distance", aggfunc="mean").reindex(MODELS)
print("=== mean edit magnitude, model x instruction ===")
print(tab.round(3).to_string())

# ---------- 1. attribution within magnitude quintiles ----------
tr = df[(split == "train") & edited]
te = df[(split == "test") & edited]
q = np.quantile(tr.normalized_edit_distance, [0.2, 0.4, 0.6, 0.8])
tr = tr.copy(); te = te.copy()
tr["magq"] = np.digitize(tr.normalized_edit_distance, q)
te["magq"] = np.digitize(te.normalized_edit_distance, q)

rows = []
for k in range(5):
    a = tr[tr.magq == k]; b = te[te.magq == k]
    if len(a) < 200 or len(b) < 60:
        continue
    det = ec.Detector("ngr", C=4.0)
    det.fit(a.abstract_clean.tolist(), a.provider.values, S[a.index.values])
    pred = det.clf.classes_[det.predict_proba(b.abstract_clean.tolist(), S[b.index.values])[1].argmax(1)]
    det_t = ec.Detector("ngr", C=4.0)
    det_t.fit(a.abstract_clean.tolist(), a.operation.values, S[a.index.values])
    pred_t = det_t.clf.classes_[det_t.predict_proba(b.abstract_clean.tolist(), S[b.index.values])[1].argmax(1)]
    rows.append(dict(quintile=f"Q{k+1}", n_train=len(a), n_test=len(b),
                     mag_lo=float(a.normalized_edit_distance.min()),
                     mag_hi=float(a.normalized_edit_distance.max()),
                     mag_mean=float(b.normalized_edit_distance.mean()),
                     attr_mf1=float(f1_score(b.provider.values, pred, average="macro")),
                     type_mf1=float(f1_score(b.operation.values, pred_t, average="macro"))))
    print(rows[-1], flush=True)
Q = pd.DataFrame(rows)
Q.to_csv(os.path.join(OUT, "attribution_by_magnitude.csv"), index=False, encoding="utf-8-sig")
R["attr_by_quintile"] = Q.attr_mf1.round(3).tolist()
print("\nattribution macro-F1 by magnitude quintile:", Q.attr_mf1.round(3).tolist())

# ---------- 2. magnitude-matched attribution ----------
lo, hi = 0.15, 0.45
a = tr[(tr.normalized_edit_distance >= lo) & (tr.normalized_edit_distance <= hi)]
b = te[(te.normalized_edit_distance >= lo) & (te.normalized_edit_distance <= hi)]
det = ec.Detector("ngr", C=4.0).fit(a.abstract_clean.tolist(), a.provider.values, S[a.index.values])
pred = det.clf.classes_[det.predict_proba(b.abstract_clean.tolist(), S[b.index.values])[1].argmax(1)]
R["attr_matched_mf1"] = float(f1_score(b.provider.values, pred, average="macro"))
R["attr_matched_n"] = [len(a), len(b)]
print(f"\nmagnitude-matched [{lo},{hi}] attribution macro-F1 = {R['attr_matched_mf1']:.3f} "
      f"(n_train={len(a)}, n_test={len(b)})")

# type classification on the same matched subset, for a like-for-like comparison
det2 = ec.Detector("ngr", C=4.0).fit(a.abstract_clean.tolist(), a.operation.values, S[a.index.values])
pred2 = det2.clf.classes_[det2.predict_proba(b.abstract_clean.tolist(), S[b.index.values])[1].argmax(1)]
R["type_matched_mf1"] = float(f1_score(b.operation.values, pred2, average="macro"))
print("type macro-F1 on the same matched subset:", round(R["type_matched_mf1"], 3))

# ---------- 3. attribution from length/punctuation only ----------
LENPUNC = ["log_chars", "n_sent", "mean_slen", "sd_slen", "cv_slen", "max_slen",
           "len_per_clause", "commas_per_sent"]
keep = [n for n in ec.STYLO_NAMES if n.startswith(("fw_", "hw_")) or n in LENPUNC
        or n in ("punct_total", "fw_share", "digits", "latin", "ttr", "hapax", "entropy")]
drop = [n for n in ec.STYLO_NAMES if n not in keep]
print(f"\nlength/punctuation subset: {len(keep)} of {len(ec.STYLO_NAMES)} features")
det3 = ec.Detector("sty", C=4.0, drop=drop).fit(
    tr.abstract_clean.tolist(), tr.provider.values, S[tr.index.values])
pred3 = det3.clf.classes_[det3.predict_proba(te.abstract_clean.tolist(), S[te.index.values])[1].argmax(1)]
R["attr_lenpunc_mf1"] = float(f1_score(te.provider.values, pred3, average="macro"))
print("attribution from length+punctuation only:", round(R["attr_lenpunc_mf1"], 3))

# lexical-only (char n-gram) is the primary model; also lexical without any length info
det4 = ec.Detector("ngr", C=4.0).fit(tr.abstract_clean.tolist(), tr.provider.values, S[tr.index.values])
pred4 = det4.clf.classes_[det4.predict_proba(te.abstract_clean.tolist(), S[te.index.values])[1].argmax(1)]
R["attr_full_mf1"] = float(f1_score(te.provider.values, pred4, average="macro"))
print("attribution, char n-gram (full):", round(R["attr_full_mf1"], 3))

# confusion among models
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(te.provider.values, pred4, labels=MODELS, normalize="true")
pd.DataFrame(cm, index=MODELS, columns=MODELS).round(3).to_csv(
    os.path.join(OUT, "attribution_confusion.csv"), encoding="utf-8-sig")
print(pd.DataFrame(cm, index=MODELS, columns=MODELS).round(3).to_string())

# ---------- 4. document-level bootstrap for the attribution vs type difference ----------
arr_a = np.load(os.path.join(OUT, "pred_type_base_ngr.npy"), allow_pickle=True)
# recompute type predictions on the full edited test set (same model as main table)
det_t = ec.Detector("ngr", C=0.25).fit(tr.abstract_clean.tolist(), tr.operation.values, S[tr.index.values])
pred_t = det_t.clf.classes_[det_t.predict_proba(te.abstract_clean.tolist(), S[te.index.values])[1].argmax(1)]
y_t = te.operation.values
y_a = te.provider.values

docs = te.source_id.values
ug = np.unique(docs)
gidx = {g: np.where(docs == g)[0] for g in ug}
rng = np.random.default_rng(0)
diffs, fa, ft = [], [], []
for _ in range(500):
    idx = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug), replace=True)])
    ma = f1_score(y_a[idx], pred4[idx], average="macro")
    mt = f1_score(y_t[idx], pred_t[idx], average="macro")
    fa.append(ma); ft.append(mt); diffs.append(ma - mt)
R["attr_vs_type_diff"] = float(R["attr_full_mf1"] - f1_score(y_t, pred_t, average="macro"))
R["attr_vs_type_diff_ci"] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
R["type_full_mf1"] = float(f1_score(y_t, pred_t, average="macro"))
print(f"\nattribution {R['attr_full_mf1']:.3f} vs type {R['type_full_mf1']:.3f}; "
      f"difference {R['attr_vs_type_diff']:.3f} [95% CI {R['attr_vs_type_diff_ci'][0]:.3f}, "
      f"{R['attr_vs_type_diff_ci'][1]:.3f}]")

# ---------- 5. per-model accuracy vs magnitude ----------
rows = []
for m in MODELS:
    g = te[te.provider == m]
    ok = (pred4[te.provider.values == m] == m).mean()
    rows.append(dict(provider=m, n=len(g), acc=float(ok),
                     mean_ed=float(g.normalized_edit_distance.mean()),
                     mean_len=float(g.char_count.mean())))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(OUT, "attribution_by_model.csv"), index=False, encoding="utf-8-sig")
print(P.round(3).to_string(index=False))

json.dump(R, open(os.path.join(OUT, "model_controlled.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("\nsaved model_controlled.json")
