import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import json, collections, re, sys, statistics as st
sys.stdout.reconfigure(encoding="utf-8")
P = DATA_FILE
rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]

print("rows", len(rows))
# 1 group structure
g = collections.Counter(r["group_id"] for r in rows)
print("\nsources/groups:", len(g), " group size dist:", collections.Counter(g.values()))
src = collections.Counter(r["source_id"] for r in rows)
print("distinct source_id:", len(src), "per-source recs:", collections.Counter(src.values()))

# 2 source prefixes
pref = collections.Counter("-".join(r["source_id"].split("-")[:2]) for r in rows)
print("\nsource prefix (first 2 tokens):", dict(pref))
pref3 = collections.Counter("-".join(r["source_id"].split("-")[:1]) for r in rows)
print("first token:", dict(pref3))

# 3 split consistency
bad = 0
split_by_group = collections.defaultdict(set)
for r in rows:
    split_by_group[r["group_id"]].add(r["group_split"])
for k, v in split_by_group.items():
    if len(v) > 1:
        bad += 1
print("\ngroups with >1 split:", bad)
print("groups per split:", collections.Counter(list(v)[0] for v in split_by_group.values()))

# 4 full crossing check
cell = collections.Counter((r["source_id"], r["operation"], r["provider"]) for r in rows)
print("\ncells:", len(cell), "dupe cells:", sum(1 for v in cell.values() if v > 1))
print("cell count dist:", collections.Counter(cell.values()))

# 5 operation x provider x level
xref = collections.Counter((r["operation"], r["provider"], r["intervention_level"]) for r in rows)
print("\noperation x provider x level combos:", len(xref))

# 6 length / edit stats per operation
print("\n%-22s %5s %7s %7s %7s %7s %7s" % ("operation", "n", "len_med", "len_p10", "len_p90", "ed_med", "numret_med"))
for op in ["original", "grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]:
    sub = [r for r in rows if r["operation"] == op]
    L = sorted(r["char_count"] for r in sub)
    E = sorted(r["normalized_edit_distance"] for r in sub)
    N = sorted(r["numeric_retention"] for r in sub)
    q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
    print("%-22s %5d %7d %7d %7d %7.3f %7.2f" % (op, len(sub), q(L, .5), q(L, .1), q(L, .9),
                                                  q(E, .5), q(N, .5)))

# 7 length change ratio (edited vs its original within group)
orig = {}
for r in rows:
    if r["operation"] == "original":
        orig[r["source_id"]] = r
print("\noriginals mapped:", len(orig))
for op in ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]:
    ratios = []
    for r in rows:
        if r["operation"] == op and r["source_id"] in orig:
            o = orig[r["source_id"]]["char_count"]
            if o:
                ratios.append(r["char_count"] / o)
    ratios.sort()
    q = lambda p: ratios[min(len(ratios) - 1, int(p * len(ratios)))]
    print(f"{op:22s} n={len(ratios)} len_ratio p10={q(.1):.2f} med={q(.5):.2f} p90={q(.9):.2f}")

# 8 duplicates / exact copies of original
same_as_orig = collections.Counter()
for r in rows:
    if r["operation"] != "original" and r["output_sha256"] == r["input_sha256"]:
        same_as_orig[r["operation"]] += 1
print("\nidentical output==input by op:", dict(same_as_orig))

# 9 dup texts
c = collections.Counter(r["abstract_clean"] for r in rows)
print("distinct abstracts:", len(c), "max dup:", max(c.values()))

# 10 chinese ratio low
low = [r for r in rows if r["chinese_ratio"] < 0.98]
print("\nchinese_ratio<0.98:", len(low), collections.Counter(r["operation"] for r in low))

# 11 sample a group
gid = rows[0]["group_id"]
print("\n=== sample group", gid, "===")
for r in rows:
    if r["group_id"] == gid:
        print(f"[{r['record_id']}] op={r['operation']:20s} prov={r['provider']:9s} len={r['char_count']:4d} ed={r['normalized_edit_distance']:.3f}")
        print("   ", r["abstract_clean"][:90].replace("\n", " "))
