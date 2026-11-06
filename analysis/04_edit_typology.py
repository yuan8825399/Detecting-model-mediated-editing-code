"""Pairwise edit typology: how does each operation actually change the text?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, re, difflib, math
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

from provenance import corpus
df = corpus.load()
OUT = PROCESSED_DIR / "pairs.pkl"

SENT = re.compile(r"[^。；！？]+[。；！？]?")
CONN = ["因此", "然而", "但是", "此外", "同时", "从而", "进而", "基于此", "在此基础",
        "值得注意", "具体而言", "换言之", "由此可见", "总之", "综上", "一方面", "另一方面",
        "不仅", "而且", "由于", "尽管", "虽然", "如果", "因而", "故而", "亦"]
BOOST = ["显著", "核心", "关键", "重要", "至关重要", "亦是", "呈现出", "凸显", "彰显",
         "深刻", "全面", "系统", "有效", "充分", "高度", "进一步", "亟需", "亟待"]
HEDGE = ["可能", "或许", "似乎", "一定程度上", "相对", "较为", "初步", "大体", "往往", "通常"]

def punc_style(t):
    fw = t.count("，") + t.count("。") + t.count("；") + t.count("：")
    hw = t.count(",") + t.count(".") + t.count(";") + t.count(":")
    return fw / (fw + hw + 1e-9)

def sents(t):
    return [s.strip() for s in SENT.findall(t) if s.strip()]

def count_any(t, pats):
    return sum(t.count(p) for p in pats)

def norm(s):
    return re.sub(r"[\s，。；：、（）()“”\"'!?！？,.;:]", "", s)

orig = df[df.operation == "original"].set_index("source_id")
rows = []
for _, r in df[df.operation != "original"].iterrows():
    o = orig.loc[r.source_id]
    a, b = o.abstract_clean, r.abstract_clean
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ins = dele = rep = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert": ins += j2 - j1
        elif tag == "delete": dele += i2 - i1
        elif tag == "replace": rep += max(i2 - i1, j2 - j1)
    # sentence-level preservation
    sa, sb = sents(a), sents(b)
    nb = set(norm(s) for s in sb)
    kept = sum(1 for s in sa if norm(s) in nb)
    na = set(norm(s) for s in sa)
    new_s = sum(1 for s in sb if norm(s) not in na)
    # char 3-gram overlap (content retention)
    def grams(t, k=3):
        t = norm(t)
        return set(t[i:i + k] for i in range(len(t) - k + 1))
    ga, gb = grams(a), grams(b)
    jac = len(ga & gb) / max(1, len(ga | gb))
    rows.append(dict(
        record_id=r.record_id, source_id=r.source_id, operation=r.operation,
        level=r.level, provider=r.provider, subfield=r.subfield, era=r.era,
        split=r.group_split,
        ed=r.normalized_edit_distance, ed_chars=len(a) and (ins + dele + rep) / len(a),
        ins_frac=ins / max(1, len(a)), del_frac=dele / max(1, len(a)), rep_frac=rep / max(1, len(a)),
        len_ratio=len(b) / max(1, len(a)),
        n_sent_a=len(sa), n_sent_b=len(sb), d_sent=len(sb) - len(sa),
        sent_kept_frac=kept / max(1, len(sa)), sent_new_frac=new_s / max(1, len(sb)),
        gram3_jac=jac,
        punc_style_a=punc_style(a), punc_style_b=punc_style(b),
        d_conn=count_any(b, CONN) - count_any(a, CONN),
        conn_rate_b=count_any(b, CONN) / max(1, len(sb)),
        n_boost_a=count_any(a, BOOST), n_boost_b=count_any(b, BOOST),
        d_boost=count_any(b, BOOST) - count_any(a, BOOST),
        n_hedge_a=count_any(a, HEDGE), d_hedge=count_any(b, HEDGE) - count_any(a, HEDGE),
        de_rate_a=a.count("的") / len(a), de_rate_b=b.count("的") / len(b),
        d_de=b.count("的") / len(b) - a.count("的") / len(a),
        mean_slen_a=len(a) / max(1, len(sa)), mean_slen_b=len(b) / max(1, len(sb)),
    ))

P = pd.DataFrame(rows)
P["d_slen"] = P.mean_slen_b - P.mean_slen_a
P.to_pickle(OUT)

pd.set_option("display.width", 200)
print("pairs:", P.shape)
print("\n=== Table 1: edit typology by operation (mean unless noted) ===")
tab = P.groupby("operation")[["ed", "ins_frac", "del_frac", "rep_frac", "len_ratio",
                             "sent_kept_frac", "sent_new_frac", "gram3_jac", "d_sent",
                             "d_conn", "d_boost", "d_de", "d_slen", "punc_style_a", "punc_style_b"]].mean()
print(tab.round(3).to_string())
print("\n=== by operation x provider: relative edit magnitude (ed) ===")
print(P.pivot_table(index="operation", columns="provider", values="ed").round(3))
print("\n=== by operation x provider: sentences kept ===")
print(P.pivot_table(index="operation", columns="provider", values="sent_kept_frac").round(3))
print("\n=== by operation x provider: boosters added ===")
print(P.pivot_table(index="operation", columns="provider", values="d_boost").round(2))
print("\n=== by operation x provider: connective delta ===")
print(P.pivot_table(index="operation", columns="provider", values="d_conn").round(2))
print("\n=== punctuation style shift (share of fullwidth) ===")
print(P.groupby("operation")[["punc_style_a", "punc_style_b"]].mean().round(4))
print("\n=== per-operation distribution of ed (quantiles) ===")
print(P.groupby("operation").ed.describe(percentiles=[.05, .25, .5, .75, .95]).round(3))
print("\n=== subfield edit magnitude ===")
print(P.pivot_table(index="subfield", columns="operation", values="ed").round(3))
print("\n=== era edit magnitude ===")
print(P.pivot_table(index="era", columns="operation", values="ed").round(3))
