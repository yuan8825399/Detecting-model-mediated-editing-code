"""Publication figures (PNG 300 dpi + PDF). Seven figures, numbered by first appearance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance.config import (  # noqa: E402
    DATA_FILE, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR,
)

import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
sys.stdout.reconfigure(encoding="utf-8")

RES = RESULTS_DIR
FIG = FIGURES_DIR
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})

OPS = ["grammar_correction", "academic_polish", "sentence_restructure", "abstract_rewrite"]
LAB = {"grammar_correction": "E1\nGrammar", "academic_polish": "E2\nPolish",
       "sentence_restructure": "E3\nRestructure", "abstract_rewrite": "E4\nRewrite"}
SHORT = {"grammar_correction": "E1 Grammar", "academic_polish": "E2 Polish",
         "sentence_restructure": "E3 Restructure", "abstract_rewrite": "E4 Rewrite"}
MODELS = ["deepseek", "doubao", "kimi", "wenxin"]
MNAME = {"deepseek": "DeepSeek", "doubao": "Doubao", "kimi": "Kimi", "wenxin": "ERNIE"}
MC = {"deepseek": "#0072B2", "doubao": "#D55E00", "kimi": "#009E73", "wenxin": "#CC79A7"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    print("saved", name)


P = pd.read_pickle(PROCESSED_DIR / "pairs.pkl")
D = json.load(open(os.path.join(RES, "descriptives.json"), encoding="utf-8"))

# ================= Figure 1: design and corpus =================
fig = plt.figure(figsize=(8.2, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.045, 0.9], wspace=1.05)
ax = fig.add_subplot(gs[0, 0])
M = P.pivot_table(index="provider", columns="operation", values="ed", aggfunc="mean").reindex(MODELS)[OPS]
im = ax.imshow(M.values, cmap="Oranges", aspect="auto", vmin=0, vmax=0.5)
ax.set_xticks(range(4)); ax.set_xticklabels([LAB[o] for o in OPS], fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels([MNAME[m] for m in MODELS])
for i in range(4):
    for j in range(4):
        v = M.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.28 else "black")
ax.set_title("a  Mean normalised edit distance\n(493 abstracts per cell)", fontsize=9, loc="left")
ax.set_xlabel("Instruction (nominal level)")
ax.set_ylabel("Editing model")
for s in ax.spines.values():
    s.set_visible(False)
cax = fig.add_subplot(gs[0, 1])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=6.5)

ax = fig.add_subplot(gs[0, 2])
comp = pd.DataFrame(D["comp_table"]).reindex(
    ["criminal", "constitutional", "civil_commercial", "administrative", "legal_theory_history", "other"])
eras = ["2000-2004", "2005-2009", "2010-2014", "2015-2020"]
bottom = np.zeros(len(comp))
ecol = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
for e, c in zip(eras, ecol):
    ax.barh(range(len(comp)), comp[e].values, left=bottom, color=c, label=e, height=0.68)
    bottom += comp[e].values
ax.set_yticks(range(len(comp)))
ax.set_yticklabels(["Criminal", "Constitutional", "Civil/commercial", "Administrative",
                    "Legal theory/history", "Other"], fontsize=8)
ax.set_xlabel("Number of source abstracts")
ax.set_title("b  Corpus composition", fontsize=9, loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=7, title="Era", title_fontsize=7)
ax.invert_yaxis()
save(fig, "fig1_design")

# ================= Figure 2: what the edits do =================
fig = plt.figure(figsize=(7.2, 5.2))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
data, pos, colors, ticks = [], [], [], []
p = 0
for o in OPS:
    for m in MODELS:
        data.append(P[(P.operation == o) & (P.provider == m)].ed.values)
        pos.append(p); colors.append(MC[m]); p += 1
    ticks.append(p - 2.5); p += 0.8
bp = ax.boxplot(data, positions=pos, widths=0.68, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("black"); patch.set_linewidth(0.5)
ax.set_xticks(ticks); ax.set_xticklabels([SHORT[o] for o in OPS])
ax.set_ylabel("Normalised edit distance"); ax.set_ylim(-0.02, 0.78)
ax.set_title("a  How much each instruction changes the text", loc="left", fontsize=9)
ax.legend(handles=[mpatches.Patch(color=MC[m], label=MNAME[m]) for m in MODELS],
          frameon=False, ncol=4, loc="upper left", fontsize=7)

ax = fig.add_subplot(gs[1, 0])
Sf = P.pivot_table(index="operation", columns="provider", values="sent_kept_frac").reindex(OPS)[MODELS]
x = np.arange(4); w = 0.2
for k, m in enumerate(MODELS):
    ax.bar(x + (k - 1.5) * w, Sf[m].values, width=w, color=MC[m], label=MNAME[m])
ax.set_xticks(x); ax.set_xticklabels(["E1", "E2", "E3", "E4"])
ax.set_ylabel("Fraction of sentences preserved")
ax.set_title("b  Sentence-level preservation", loc="left", fontsize=9)
ax.set_ylim(0, 1.18)
ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper right")

ax = fig.add_subplot(gs[1, 1])
cats = ["All edits", "E1 only", "E2–E4"]
keys = ["decomp_all", "decomp_E1", "decomp_E2_E4"]
op_v = [D[k]["eta2_operation"] for k in keys]
md_v = [D[k]["eta2_provider"] for k in keys]
sr_v = [D[k]["eta2_source_id"] for k in keys]
x = np.arange(3); w = 0.26
ax.bar(x - w, op_v, w, label="Instruction", color="#4C72B0")
ax.bar(x, md_v, w, label="Model", color="#C44E52")
ax.bar(x + w, sr_v, w, label="Document", color="#999999")
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("η² (variance explained)"); ax.set_ylim(0, 0.75)
ax.set_title("c  What drives edit magnitude", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=7)
save(fig, "fig2_edits")

# ================= Figure 3: detectability =================
dose = pd.read_csv(os.path.join(RES, "dose_response.csv"), encoding="utf-8-sig")
pc = pd.read_csv(os.path.join(RES, "per_condition_ceiling.csv"), encoding="utf-8-sig")
tm = pd.read_csv(os.path.join(RES, "per_type_model_detectability.csv"), encoding="utf-8-sig")

fig = plt.figure(figsize=(7.4, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.42)

ax = fig.add_subplot(gs[0, 0])
x = dose.mean_ed.values
ax.errorbar(x, dose.tpr, yerr=[dose.tpr - dose.lo, dose.hi - dose.tpr], fmt="o-",
            color="#C44E52", lw=1.5, ms=5, capsize=3, ecolor="#C44E52", elinewidth=1)
ax.plot(x, dose.tpr_at_1pct, "s--", color="#4C72B0", lw=1.1, ms=4)
for xi, ni, yi in zip(x, dose.n, dose.tpr):
    ax.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=5.8, color="#555555")
ax.set_xlabel("Edit magnitude (normalised edit distance)")
ax.set_ylabel("Detection rate")
ax.set_ylim(-0.03, 1.06)
ax.set_title("a  Detection rises with edit magnitude", loc="left", fontsize=9)
ax.legend(handles=[Line2D([], [], color="#C44E52", marker="o", label="TPR, nominal 5% calibration FPR"),
                   Line2D([], [], color="#4C72B0", ls="--", marker="s", label="TPR, nominal 1% calibration FPR")],
          frameon=False, fontsize=6.8, loc="lower right", bbox_to_anchor=(1.0, 0.12))
ax.text(0.98, 0.02, "Spearman ρ = 0.64", transform=ax.transAxes, fontsize=6.6,
        color="#444444", ha="right")

ax = fig.add_subplot(gs[0, 1])
d = pc[pc.operation != "pooled"].reset_index(drop=True)
x = np.arange(len(d))
ax.bar(x, d.tpr, color="#8172B2", width=0.45)
for i, r in d.iterrows():
    ax.plot([i - 0.34, i + 0.34], [r.ceiling, r.ceiling], color="black", lw=1.6)
    ax.text(i, r.ceiling + 0.025, f"{r.ceiling:.2f}", ha="center", fontsize=6.5)
    ax.text(i, r.tpr / 2, f"{r.normalised_tpr:.2f}", ha="center", fontsize=7.5,
            color="white", fontweight="bold")
ax.text(0.02, 0.98, "white number = TPR / non-identical fraction",
        transform=ax.transAxes, fontsize=5.8, color="#444444", va="top")
ax.set_xticks(x); ax.set_xticklabels(["E1", "E2", "E3", "E4"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Rate")
ax.set_title("b  Detection and non-identical fraction", loc="left", fontsize=9)

ax = fig.add_subplot(gs[1, :])
Mx = tm.pivot_table(index="operation", columns="provider", values="tpr").reindex(OPS)[MODELS]
im = ax.imshow(Mx.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([MNAME[m] for m in MODELS], fontsize=7.5)
ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[o] for o in OPS], fontsize=7.5)
for i in range(4):
    for j in range(4):
        v = Mx.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="black" if 0.25 < v < 0.85 else "white")
ax.set_title("c  Detection at calibration-fixed threshold, by instruction and model", loc="left", fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("TPR (calibration-fixed)", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "fig3_detectability")

# ================= Figure 4: type confusion =================
arr = np.load(os.path.join(RES, "pred_type_base_ngr.npy"), allow_pickle=True)
preds = arr[:, 1].astype(str)
from provenance import corpus
df = corpus.load()
te_ed = df[(df.group_split == "test") & (df.intervention_present == 1)]
C = np.zeros((4, 4))
for t, pr in zip(te_ed.operation.values, preds):
    C[OPS.index(t), OPS.index(pr)] += 1
C = C / C.sum(1, keepdims=True)

fig, ax = plt.subplots(figsize=(3.4, 3.0))
im = ax.imshow(C, cmap="Blues", vmin=0, vmax=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(["E1", "E2", "E3", "E4"], fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(["E1", "E2", "E3", "E4"], fontsize=8)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if C[i, j] > 0.45 else "black")
ax.set_xlabel("Predicted instruction"); ax.set_ylabel("True instruction")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
save(fig, "fig4_type_confusion")

# ================= Figure 5: model attribution under magnitude control =================
Q = pd.read_csv(os.path.join(RES, "attribution_by_magnitude.csv"), encoding="utf-8-sig")
CM = pd.read_csv(os.path.join(RES, "attribution_confusion.csv"), index_col=0, encoding="utf-8-sig")[MODELS]
MCtl = json.load(open(os.path.join(RES, "model_controlled.json"), encoding="utf-8"))

fig = plt.figure(figsize=(7.2, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.34)

ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(Q)); w = 0.36
ax.bar(x - w / 2, Q.attr_mf1, w, label="Model attribution", color="#C44E52")
ax.bar(x + w / 2, Q.type_mf1, w, label="Instruction type", color="#4C72B0")
ax.axhline(0.25, color="black", ls=":", lw=0.8)
ax.text(len(Q) - 0.45, 0.265, "chance", fontsize=6.5, ha="right")
ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}\n{r:.2f}" for i, r in enumerate(Q.mag_mean)], fontsize=7)
ax.set_xlabel("Edit-magnitude quintile (mean normalised edit distance)")
ax.set_ylabel("macro-F1")
ax.set_ylim(0, 0.78)
ax.set_title("a  Both tasks are unrecoverable for near-verbatim edits", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=6.6, loc="upper left", bbox_to_anchor=(0.0, 1.02))
ax.annotate(f"magnitude-matched [0.15, 0.45]:\nattribution {MCtl['attr_matched_mf1']:.3f}"
            f" vs type {MCtl['type_matched_mf1']:.3f}",
            xy=(3.9, 0.76), fontsize=6.3, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F4F4F4", ec="#BBBBBB", lw=0.6))

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(CM.values, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([MNAME[m] for m in MODELS], rotation=30, ha="right", fontsize=7.5)
ax.set_yticks(range(4)); ax.set_yticklabels([MNAME[m] for m in MODELS], fontsize=7.5)
for i in range(4):
    for j in range(4):
        v = CM.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.45 else "black")
ax.set_xlabel("Predicted model"); ax.set_ylabel("True model")
ax.set_title("b  Attribution confusion (test, all edits)", loc="left", fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)
save(fig, "fig5_attribution")

# ================= Figure 6: cross-model transfer =================
cm_auc = pd.read_csv(os.path.join(RES, "cross_model_auc.csv"), index_col=0, encoding="utf-8-sig")[MODELS]
tr = pd.read_csv(os.path.join(RES, "transfer_results.csv"), encoding="utf-8-sig")

fig = plt.figure(figsize=(7.2, 4.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.34)
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(cm_auc.values, cmap="Blues", vmin=0.5, vmax=0.9, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([MNAME[m] for m in MODELS], rotation=30, ha="right", fontsize=7.5)
ax.set_yticks(range(4)); ax.set_yticklabels([MNAME[m] for m in MODELS], fontsize=7.5)
for i in range(4):
    for j in range(4):
        v = cm_auc.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.78 else "black",
                fontweight="bold" if i == j else "normal")
ax.set_xlabel("Test model"); ax.set_ylabel("Training model")
ax.set_title("a  Cross-model transfer (AUC)", loc="left", fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)

ax = fig.add_subplot(gs[0, 1])
lomo = tr[(tr.protocol == "LOMO") & (tr.task == "presence")]
loso = tr[(tr.protocol == "LOSO") & (tr.task == "presence")]
baselines = pd.read_csv(RES / "transfer_baselines.csv")
in_domain = float(baselines[baselines.task == "presence"].auc.iloc[0])
gvals = [("In-domain\n(fused)", in_domain, "#4C72B0")]
for _, r in lomo.iterrows():
    gvals.append((f"LOMO\n{MNAME[r.heldout]}", r.auc, "#C44E52"))
gvals.append(("LOSO\n(mean)", loso.auc.mean(), "#55A868"))
xs = np.arange(len(gvals))
ax.bar(xs, [g[1] for g in gvals], color=[g[2] for g in gvals], width=0.68)
ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in gvals], fontsize=6.6)
ax.set_ylabel("AUC (presence detection)")
ax.set_ylim(0.5, 0.95)
ax.axhline(in_domain, color="#4C72B0", ls="--", lw=0.8)
ax.set_title("b  Model hold-out vs. domain hold-out", loc="left", fontsize=9)
for xx, g in zip(xs, gvals):
    ax.text(xx, g[1] + 0.020, f"{g[1]:.3f}", ha="center", fontsize=7)
save(fig, "fig6_reliability")

# ================= Figure 7: paired discrimination + human FPR =================
pr = pd.read_csv(os.path.join(RES, "paired_discrimination.csv"), encoding="utf-8-sig")
fpr = pd.read_csv(os.path.join(RES, "fpr_strata.csv"), encoding="utf-8-sig")
fpr_e = fpr[fpr.stratum == "era"].reset_index(drop=True)

fig = plt.figure(figsize=(7.2, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.34)
ax = fig.add_subplot(gs[0, 0])
W = pr.pivot_table(index="operation", columns="provider", values="win").reindex(OPS)[MODELS]
im = ax.imshow(W.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([MNAME[m] for m in MODELS], rotation=30, ha="right", fontsize=7.5)
ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[o] for o in OPS], fontsize=7.5)
for i in range(4):
    for j in range(4):
        v = W.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="black" if 0.3 < v < 0.85 else "white")
ax.set_title("a  Paired discrimination P[score(edit) > score(source)]", loc="left", fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)

ax = fig.add_subplot(gs[0, 1])
x = np.arange(len(fpr_e))
ax.bar(x, fpr_e.fpr, color="#8172B2", width=0.6,
       yerr=[fpr_e.fpr - fpr_e.lo, fpr_e.hi - fpr_e.fpr], capsize=4,
       error_kw=dict(lw=0.9, ecolor="#444444"))
ax.set_xticks(x)
ax.set_xticklabels([str(i).replace("-", "–") for i in fpr_e.level], fontsize=7.5)
ax.set_ylabel("False-positive rate on human text\n(threshold fixed on calibration set)")
ax.set_ylim(0, 0.50)
for i, r in fpr_e.iterrows():
    ax.text(i, r.hi + 0.012, f"{r.fpr:.2f}\n(n={int(r.n)})", ha="center", fontsize=6.5)
ax.set_title("b  False positives on human abstracts by era", loc="left", fontsize=9)
save(fig, "fig7_fpr")
print("all figures written")
