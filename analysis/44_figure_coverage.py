"""Figure 9 (three panels): empirical coverage evaluation."""

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
sys.stdout.reconfigure(encoding="utf-8")

RES = RESULTS_DIR
FIG = FIGURES_DIR
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})

V = json.load(open(os.path.join(RES, "palcp_verification.json"), encoding="utf-8"))
C = pd.DataFrame(V["coverage_curve"])
S = pd.DataFrame(V["shift"])

fig = plt.figure(figsize=(7.4, 2.85))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.85], wspace=0.42)

# (a) coverage vs nominal
ax = fig.add_subplot(gs[0, 0])
for var, col, mk, lab in [("class-conditional", "#C44E52", "o", "Class-conditional"),
                          ("marginal", "#4C72B0", "s", "Marginal")]:
    d = C[C.variant == var].sort_values("nominal")
    ax.errorbar(d.nominal, d.coverage, yerr=[d.coverage - d.lo, d.hi - d.coverage],
                fmt=mk + "-", color=col, ms=4, lw=1.2, capsize=2.5, label=lab)
ax.plot([0.69, 1.0], [0.69, 1.0], "--", color="black", lw=0.7)
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_xlim(0.69, 1.005); ax.set_ylim(0.65, 1.005)
ax.set_title("a  Overall coverage", loc="left", fontsize=9)
ax.legend(frameon=False, loc="upper left", fontsize=6.4)

# (b) human-class coverage
ax = fig.add_subplot(gs[0, 1])
d = C[C.variant == "class-conditional"].sort_values("nominal")
ax.plot(d.nominal, d.cov_human, "o-", color="#009E73", ms=4, lw=1.2, label="Class-conditional")
d2 = C[C.variant == "marginal"].sort_values("nominal")
ax.plot(d2.nominal, d2.cov_human, "s--", color="#4C72B0", ms=4, lw=1.2, label="Marginal")
ax.plot([0.69, 1.0], [0.69, 1.0], ":", color="black", lw=0.7)
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Coverage of human originals")
ax.set_xlim(0.69, 1.005); ax.set_ylim(-0.03, 1.03)
ax.set_title("b  Coverage of the human class", loc="left", fontsize=9)
ax.legend(frameon=False, loc="upper left", fontsize=6.4)
ax.annotate("never covers\na human text", xy=(0.85, 0.01), xytext=(0.72, 0.32),
            fontsize=6.0, arrowprops=dict(arrowstyle="->", lw=0.6, color="#444444"))

# (c) model shift
ax = fig.add_subplot(gs[0, 2])
x = np.arange(len(S))
ax.bar(x, S.coverage, color="#8172B2", width=0.6)
ax.axhline(0.90, color="black", ls=":", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(["Deep\nSeek", "Dou\nbao", "Kimi", "ERNIE"], fontsize=6.8)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Coverage")
ax.set_title("c  Model held out", loc="left", fontsize=9)
for i, v in enumerate(S.coverage):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.4)
ax.text(-0.45, 0.925, "nominal", fontsize=6.0, ha="left")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig9_coverage.{ext}"), dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
plt.close(fig)
print("saved fig9_coverage (3 panels)")
