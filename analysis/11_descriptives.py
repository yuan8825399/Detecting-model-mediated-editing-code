"""Compute every descriptive statistic cited in the manuscript; write tables to markdown/CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, json, os
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
from provenance import corpus

OUT = RESULTS_DIR
os.makedirs(OUT, exist_ok=True)
df = corpus.load().reset_index(drop=True)
P = pd.read_pickle(PROCESSED_DIR / "pairs.pkl")
R = {}

OPS = ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]
LAB = {"grammar_correction": "E1 grammar", "academic_polish": "E2 polish",
       "sentence_restructure": "E3 restructure", "abstract_rewrite": "E4 rewrite"}
MODELS = ["deepseek", "doubao", "kimi", "wenxin"]

# ---------- basics ----------
R["n_sources"] = int(df.source_id.nunique())
R["n_records"] = int(len(df))
R["n_edited"] = int((df.intervention_present == 1).sum())
R["len_mean"] = float(df[df.operation == "original"].char_count.mean())
R["len_sd"] = float(df[df.operation == "original"].char_count.std())
R["len_min"] = int(df[df.operation == "original"].char_count.min())
R["len_max"] = int(df[df.operation == "original"].char_count.max())
R["subfield"] = df[df.operation == "original"].subfield.value_counts().to_dict()
R["split"] = df.groupby("group_split").source_id.nunique().to_dict()

# ---------- Table 1: edit typology ----------
t1 = P.groupby("operation").agg(
    n=("ed", "size"), ed=("ed", "mean"), ed_med=("ed", "median"),
    ins=("ins_frac", "mean"), dele=("del_frac", "mean"), rep=("rep_frac", "mean"),
    len_ratio=("len_ratio", "mean"), sent_kept=("sent_kept_frac", "mean"),
    gram3=("gram3_jac", "mean"), d_conn=("d_conn", "mean"), d_boost=("d_boost", "mean"),
    d_slen=("d_slen", "mean"), punc_b=("punc_style_b", "mean")).reindex(OPS)
R["table1"] = t1.round(3).to_dict(orient="index")

t1m = t1.copy()
t1m.index = [LAB[i] for i in t1m.index]
md = ["| Operation | n | Mean ED | Median ED | Ins. | Del. | Sub. | Length ratio | Sentences preserved | 3-gram Jaccard | Δ connectives | Δ boosters | Δ mean sentence length |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in t1m.iterrows():
    md.append(f"| {i} | {int(r.n)} | {r.ed:.3f} | {r.ed_med:.3f} | {r.ins*100:.1f}% | {r.dele*100:.1f}% | "
              f"{r.rep*100:.1f}% | {r.len_ratio:.3f} | {r.sent_kept*100:.1f}% | {r.gram3:.3f} | "
              f"{r.d_conn:+.2f} | {r.d_boost:+.2f} | {r.d_slen:+.1f} |")
open(os.path.join(OUT, "table1_typology.md"), "w", encoding="utf-8").write("\n".join(md))

# ---------- Table 2: operation x model ----------
piv_ed = P.pivot_table(index="operation", columns="provider", values="ed", aggfunc="mean").reindex(OPS)[MODELS]
piv_med = P.pivot_table(index="operation", columns="provider", values="ed", aggfunc="median").reindex(OPS)[MODELS]
eff = df[df.intervention_present == 1].pivot_table(index="operation", columns="provider",
                                                   values="effective", aggfunc="mean").reindex(OPS)[MODELS]
piv_len = P.pivot_table(index="operation", columns="provider", values="len_ratio", aggfunc="mean").reindex(OPS)[MODELS]
R["table2_ed"] = piv_ed.round(3).to_dict()
R["table2_eff"] = eff.round(3).to_dict()
md = ["| Operation | " + " | ".join(f"{m.capitalize()} ED (median)" for m in MODELS) + " |",
      "|---|" + "---|" * len(MODELS)]
for i in OPS:
    md.append(f"| {LAB[i]} | " + " | ".join(f"{piv_ed.loc[i,m]:.3f} ({piv_med.loc[i,m]:.3f})" for m in MODELS) + " |")
open(os.path.join(OUT, "table2_op_model.md"), "w", encoding="utf-8").write("\n".join(md))

md = ["| Operation | " + " | ".join(f"{m.capitalize()}" for m in MODELS) + " |",
      "|---|" + "---|" * len(MODELS)]
for i in OPS:
    md.append(f"| {LAB[i]} | " + " | ".join(f"{eff.loc[i,m]*100:.1f}%" for m in MODELS) + " |")
open(os.path.join(OUT, "table3_effective.md"), "w", encoding="utf-8").write("\n".join(md))

# ---------- verbatim / minimal-edit rates ----------
verb = df[df.intervention_present == 1].groupby("operation").identical.mean().reindex(OPS)
small = df[df.intervention_present == 1].assign(small=(df.edit_chars < 5)).groupby("operation").small.mean().reindex(OPS)
R["verbatim_rate"] = verb.round(4).to_dict()
R["small_edit_rate"] = small.round(4).to_dict()
R["verb_at_all"] = float(df[df.intervention_present == 1].identical.mean())
print("verbatim:", verb.round(3).to_dict(), "overall", round(R["verb_at_all"], 4))
print("<5 chars:", small.round(3).to_dict())

# provider-level verbatim rates in E1
v1 = df[(df.operation == "grammar_correction")].groupby("provider").identical.mean()
R["verb_E1_by_model"] = v1.round(4).to_dict()
print("E1 verbatim by model:", v1.round(3).to_dict())

# ---------- variance decomposition ----------
ed = df[df.intervention_present == 1].copy()


def decompose(d, tag):
    grand = d.normalized_edit_distance.mean()
    ss_tot = ((d.normalized_edit_distance - grand) ** 2).sum()
    out = {"n": len(d), "grand_mean": float(grand)}
    for fac in ["operation", "provider", "source_id"]:
        ss = d.groupby(fac).apply(
            lambda g: len(g) * (g.normalized_edit_distance.mean() - grand) ** 2,
            include_groups=False).sum()
        out[f"eta2_{fac}"] = float(ss / ss_tot)
    R[f"decomp_{tag}"] = out
    print(tag, {k: round(v, 3) for k, v in out.items()})
    return out


decompose(ed, "all")
decompose(ed[ed.operation == "grammar_correction"], "E1")
decompose(ed[ed.operation != "grammar_correction"], "E2_E4")
for opn in OPS[1:]:
    decompose(ed[ed.operation == opn], opn)
R["eta2_op"] = R["decomp_all"]["eta2_operation"]
R["eta2_model"] = R["decomp_all"]["eta2_provider"]

# quantify: model spread vs operation spread
mean_ed_model = ed.groupby("provider").normalized_edit_distance.mean()
mean_ed_op = ed.groupby("operation").normalized_edit_distance.mean()
R["model_spread_ratio"] = float(mean_ed_model.max() / mean_ed_model.min())
R["op_spread_ratio_E2E4"] = float(mean_ed_op[["academic_polish", "sentence_restructure", "abstract_rewrite"]].max() /
                                  mean_ed_op[["academic_polish", "sentence_restructure", "abstract_rewrite"]].min())
R["mean_ed_by_model"] = mean_ed_model.round(3).to_dict()
print("model mean ed:", mean_ed_model.round(3).to_dict())

# Kruskal-Wallis tests
kw_op = stats.kruskal(*[g.normalized_edit_distance.values for _, g in ed.groupby("operation")])
kw_md = stats.kruskal(*[g.normalized_edit_distance.values for _, g in ed.groupby("provider")])
R["kw_op"] = dict(H=float(kw_op.statistic), p=float(kw_op.pvalue))
R["kw_model"] = dict(H=float(kw_md.statistic), p=float(kw_md.pvalue))
print("KW op", kw_op.statistic, kw_op.pvalue, "| KW model", kw_md.statistic, kw_md.pvalue)

# E2 vs E3 vs E4 pairwise Wilcoxon on paired documents (per-source means)
ps = ed.groupby(["source_id", "operation"]).normalized_edit_distance.mean().unstack()
for a, b in [("academic_polish", "sentence_restructure"), ("sentence_restructure", "abstract_rewrite"),
             ("academic_polish", "abstract_rewrite")]:
    w = stats.wilcoxon(ps[a], ps[b])
    d = (ps[a] - ps[b]).mean()
    R[f"wilcoxon_{a}_vs_{b}"] = dict(stat=float(w.statistic), p=float(w.pvalue), mean_diff=float(d))
    print(f"{a} vs {b}: mean diff {d:+.3f}, p={w.pvalue:.2e}")

# ---------- E1 identity: is it model-dependent? ----------
ct = pd.crosstab(df[df.operation == "grammar_correction"].provider,
                 df[df.operation == "grammar_correction"].identical)
chi = stats.chi2_contingency(ct)
R["chi2_E1_identity"] = dict(chi2=float(chi.statistic), p=float(chi.pvalue), dof=int(chi.dof))
print("chi2 E1 identity by model: chi2=%.1f p=%.2e" % (chi.statistic, chi.pvalue))

# ---------- sentence preservation / length by op ----------
print("\ntable1 saved")

# ---------- corpus composition table ----------
comp = pd.crosstab(df[df.operation == "original"].subfield, df[df.operation == "original"].era)
R["comp_table"] = comp.to_dict()
md = ["| Subfield | 2000–2004 | 2005–2009 | 2010–2014 | 2015–2020 | Total |",
      "|---|---|---|---|---|---|"]
for s in comp.index:
    row = comp.loc[s]
    md.append(f"| {s.replace('_',' ').title()} | {row['2000-2004']} | {row['2005-2009']} | {row['2010-2014']} | {row['2015-2020']} | {row.sum()} |")
tot = comp.sum()
md.append(f"| **Total** | {tot['2000-2004']} | {tot['2005-2009']} | {tot['2010-2014']} | {tot['2015-2020']} | {tot.sum()} |")
open(os.path.join(OUT, "table0_composition.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md))

json.dump(R, open(os.path.join(OUT, "descriptives.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nwritten descriptives.json")
