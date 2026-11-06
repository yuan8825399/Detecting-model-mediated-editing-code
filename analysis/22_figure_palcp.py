"""Figure 8: behaviour of the PAL-CP audit method."""

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
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.linewidth": 0.8,
})

D = np.load(os.path.join(RES, "palcp_test.npz"), allow_pickle=True)
inc_p, inc_a, y, op, g = D["inc_p"], D["inc_a"], D["y"], D["op"], D["g"]
undec = inc_p & inc_a
dec_present = inc_p & ~inc_a
dec_absent = ~inc_p & inc_a
empty = ~inc_p & ~inc_a

fig = plt.figure(figsize=(7.2, 3.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.36)

# (a) decisiveness and coverage against edit magnitude
ax = fig.add_subplot(gs[0, 0])
bins = [-1e-9, 1e-9, 0.05, 0.15, 0.30, 10]
labs = ["0", "≤0.05", "0.05–0.15", "0.15–0.30", ">0.30"]
b = np.asarray(pd.cut(g, bins, labels=labs).astype(str))
x = np.arange(len(labs))
dec = [float(( (dec_present | dec_absent) )[b == l].mean()) for l in labs]
cov = [float(np.where(y == 1, inc_p, inc_a)[b == l].mean()) for l in labs]
ax.bar(x, dec, color="#8172B2", width=0.55, label="Decisive verdict")
ax.plot(x, cov, "o-", color="#C44E52", lw=1.5, ms=5, label="Coverage (true label in set)")
ax.axhline(0.90, color="grey", ls=":", lw=0.9)
ax.text(0.05, 0.925, "nominal 90% coverage", fontsize=6.3, ha="left", color="#555555")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7)
ax.set_xlabel("Edit magnitude")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1.12)
ax.set_title("a  Decisiveness tracks edit magnitude", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.20))

# (b) verdict composition per true condition
ax = fig.add_subplot(gs[0, 1])
cond = [("Human", y == 0)] + [(n, op == n) for n in
                              ["grammar_correction", "academic_polish",
                               "sentence_restructure", "abstract_rewrite"]]
names = ["Human", "E1", "E2", "E3", "E4"]
pres_only = [float(dec_present[m].mean()) for _, m in cond]
abs_only = [float(dec_absent[m].mean()) for _, m in cond]
und = [float(undec[m].mean()) for _, m in cond]
x = np.arange(len(cond))
ax.bar(x, pres_only, color="#4C72B0", width=0.62, label="“present” (decisive)")
ax.bar(x, abs_only, bottom=pres_only, color="#DD8452", width=0.62, label="“absent” (decisive)")
ax.bar(x, und, bottom=np.array(pres_only) + np.array(abs_only),
       color="#CCCCCC", width=0.62, label="undecidable")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Share of records")
ax.set_ylim(0, 1.0)
ax.set_title("b  Verdict composition by true condition", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=6.4, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, -0.16))
ax.annotate("34.6% of E1 texts are\ndecisively called “absent”",
            xy=(1, 0.30), xytext=(1.55, 0.55), fontsize=6.4,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#444444"))

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig8_palcp.{ext}"), dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
plt.close(fig)
print("saved fig8_palcp")
