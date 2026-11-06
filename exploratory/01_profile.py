import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import json, collections, sys, os
sys.stdout.reconfigure(encoding="utf-8")
P = DATA_FILE

rows = []
bad = 0
with open(P, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            bad += 1
            print("BAD LINE", i, e)

print("total rows:", len(rows), "bad:", bad)
keys = collections.Counter()
for r in rows:
    keys.update(r.keys())
print("\n--- keys (count of presence) ---")
for k, v in keys.most_common():
    print(f"{k:28s} {v}")

def dist(field, limit=40):
    c = collections.Counter()
    for r in rows:
        v = r.get(field, "<MISSING>")
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)[:60]
        c[v] += 1
    print(f"\n--- {field} ({len(c)} distinct) ---")
    for v, n in c.most_common(limit):
        print(f"  {str(v)[:70]:70s} {n}")

for f_ in ["corpus_id", "operation", "intervention_present", "intervention_level",
           "provider", "model_name", "model_snapshot", "prompt_id", "group_split",
           "evaluation_role", "quality_status", "source_year", "normalization_applied"]:
    dist(f_)

# numeric summaries
print("\n--- numeric fields ---")
for f_ in ["char_count", "chinese_ratio", "normalized_edit_distance", "numeric_retention"]:
    vals = [r[f_] for r in rows if isinstance(r.get(f_), (int, float))]
    if vals:
        vals.sort()
        n = len(vals)
        print(f"{f_}: n={n} min={vals[0]:.4f} p25={vals[n//4]:.4f} med={vals[n//2]:.4f} p75={vals[3*n//4]:.4f} max={vals[-1]:.4f} mean={sum(vals)/n:.4f}")
