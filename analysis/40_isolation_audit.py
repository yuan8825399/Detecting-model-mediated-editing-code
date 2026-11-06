"""Source-isolation audit.

Mechanically verifies, for every experimental protocol reported in the paper, that the
document sets used for fitting and evaluation are pairwise disjoint, that no source
contributes records to two roles, and that no exact duplicate document
crosses a split boundary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json, hashlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
df = corpus.load().reset_index(drop=True)

split = df.group_split.values
src = df.source_id.values
grp = df.group_id.values
prov = df.provider.values
sub = df.subfield.values
era = df.era.values
op = df.operation.values
is_orig = (df.operation == "original").values
edited = ~is_orig

rows = []


def audit(name, train_idx, test_idx, calib_idx=None, extra=""):
    tr = set(src[train_idx])
    te = set(src[test_idx])
    ca = set(src[calib_idx]) if calib_idx is not None else set()
    overlap_tt = tr & te
    overlap_tc = tr & ca
    overlap_ct = ca & te
    ok = not (overlap_tt or overlap_tc or overlap_ct)
    rows.append(dict(
        protocol=name,
        n_train_docs=len(tr), n_train_records=len(train_idx),
        n_test_docs=len(te), n_test_records=len(test_idx),
        n_calib_docs=len(ca), n_calib_records=0 if calib_idx is None else len(calib_idx),
        doc_overlap_train_test=len(overlap_tt),
        doc_overlap_train_calib=len(overlap_tc),
        doc_overlap_calib_test=len(overlap_ct),
        isolated=ok, note=extra))
    if not ok:
        print("!! OVERLAP", name, list(overlap_tt)[:5], list(overlap_tc)[:5], list(overlap_ct)[:5])
    return ok


# ---------- 0. split integrity ----------
per_group = df.groupby("group_id").group_split.nunique()
per_source = df.groupby("source_id").group_split.nunique()
print("group_id with >1 split:", int((per_group > 1).sum()))
print("source_id with >1 split:", int((per_source > 1).sum()))
print("records per source:", sorted(df.groupby("source_id").size().unique()))
print("source_id <-> group_id one-to-one:",
      df.groupby("source_id").group_id.nunique().max() == 1 and
      df.groupby("group_id").source_id.nunique().max() == 1)

# duplicate documents across splits (same title hash / same cleaned abstract)
dup_title = df.groupby("title_sha256").source_id.nunique()
dup_abs = df.groupby("abstract_clean").source_id.nunique()
print("title hashes shared by >1 source:", int((dup_title > 1).sum()))
print("identical abstracts across different sources:", int((dup_abs > 1).sum()))

# same abstract appearing in two splits (should be impossible if source-disjoint)
tsp = df.groupby("abstract_clean").group_split.nunique()
print("abstracts spanning >1 split:", int((tsp > 1).sum()))

# ---------- 1. main tasks ----------
itr = np.where(split == "train")[0]
ite = np.where(split == "test")[0]
iva = np.where(split == "dev")[0]
icb = np.where(split == "calibration")[0]
audit("T1-T4 main tasks (train vs test)", itr, ite)
audit("Hyperparameter selection (train vs dev)", itr, iva)
audit("Threshold + conformal calibration (train vs calibration)", itr, icb)
audit("Calibration vs test", icb, ite)

# ---------- 2. leave-one-model-out ----------
for m in ["deepseek", "doubao", "kimi", "wenxin"]:
    a = np.where((split == "train") & (prov != m))[0]
    b = np.where((split == "test") & ((prov == m) | is_orig))[0]
    audit(f"LOMO hold-out {m}", a, b, extra="human originals on both sides are distinct documents")

# ---------- 3. cross-model matrix ----------
for m1 in ["deepseek", "doubao", "kimi", "wenxin"]:
    a = np.where((split == "train") & (is_orig | (prov == m1)))[0]
    for m2 in ["deepseek", "doubao", "kimi", "wenxin"]:
        b = np.where((split == "test") & (is_orig | (prov == m2)))[0]
        audit(f"Cross-model train {m1} -> test {m2}", a, b)

# ---------- 4. leave-one-subfield-out ----------
for s in sorted(set(sub)):
    a = np.where((split == "train") & (sub != s))[0]
    b = np.where((split == "test") & (sub == s))[0]
    if len(b) == 0:
        continue
    audit(f"LOSO subfield {s}", a, b)

# ---------- 5. cross-era ----------
for e1 in sorted(set(era)):
    a = np.where((split != "test") & (era == e1))[0]
    for e2 in sorted(set(era)):
        b = np.where((split == "test") & (era == e2))[0]
        audit(f"Cross-era train {e1} -> test {e2}", a, b,
              extra="train side uses all non-test partitions")

# ---------- 6. cross-intensity ----------
light = ["grammar_correction", "academic_polish"]
heavy = ["sentence_restructure", "abstract_rewrite"]
for nm, src_ops, tgt_ops in [("light->heavy", light, heavy), ("heavy->light", heavy, light),
                             ("E1->E4", ["grammar_correction"], ["abstract_rewrite"]),
                             ("E2->E3", ["academic_polish"], ["sentence_restructure"])]:
    a = np.where((split == "train") & (is_orig | np.isin(op, src_ops)))[0]
    b = np.where((split == "test") & (is_orig | np.isin(op, tgt_ops)))[0]
    audit(f"Cross-intensity {nm}", a, b)

# ---------- 7. PAL-CP ----------
a = np.where(split == "train")[0]
audit("PAL-CP lattice + conformal (train vs calibration vs test)", a, ite, icb)

A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "isolation_audit.csv"), index=False, encoding="utf-8-sig")

print("\n=== isolation audit summary ===")
print(f"protocols audited            : {len(A)}")
print(f"protocols with any overlap   : {int((~A.isolated).sum())}")
print(f"max doc overlap train/test   : {int(A.doc_overlap_train_test.max())}")
print(f"max doc overlap train/calib  : {int(A.doc_overlap_train_calib.max())}")
print(f"max doc overlap calib/test   : {int(A.doc_overlap_calib_test.max())}")
print("\nsmallest test side:")
print(A.nsmallest(4, "n_test_docs")[["protocol", "n_train_docs", "n_test_docs"]].to_string(index=False))

# ---------- 8. duplicate-content check inside test ----------
texts = df.abstract_clean.values
print("\nidentical edited outputs within the test partition:",
      int(df[df.group_split == "test"].groupby(["abstract_clean"]).size().gt(1).sum()))

summary = dict(n_protocols=int(len(A)), n_with_overlap=int((~A.isolated).sum()),
               max_overlap_tt=int(A.doc_overlap_train_test.max()),
               max_overlap_tc=int(A.doc_overlap_train_calib.max()),
               max_overlap_ct=int(A.doc_overlap_calib_test.max()),
               groups_multi_split=int((per_group > 1).sum()),
               sources_multi_split=int((per_source > 1).sum()),
               duplicate_titles=int((dup_title > 1).sum()),
               duplicate_abstracts=int((dup_abs > 1).sum()),
               abstracts_spanning_splits=int((tsp > 1).sum()))
json.dump(summary, open(os.path.join(OUT, "isolation_audit.json"), "w", encoding="utf-8"),
          indent=1)
print("\nsaved isolation_audit.csv / .json")
