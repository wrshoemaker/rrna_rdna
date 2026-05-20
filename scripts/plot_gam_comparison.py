import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config


sens_summary_path = '%sgam_results/03_sensitivity_summary.csv' % config.data_directory


#def plot_sensitivity_heatmap():



# =============================================================================
# Per-ASV Sensitivity Heatmap
#
# Colour metric: within-predictor z-score of mean_abs_deriv
#   - Removes the 300x scale difference between salinity and total phosphorus
#   - Answers: for each predictor, which ASVs are most responsive?
#   - Diverging palette (blue=below average, red=above average) per predictor
#
# Three panels: one per data type (DNA / RNA / RNA:DNA)
# =============================================================================


#DATA_PATH = '%sgam_results/02_sensitivity_per_triplet.csv' % config.data_directory
#OUT_PATH  = "%sgam_results/fig_per_asv_sensitivity_heatmap.png" % config.data_directory


# =============================================================================
# Directional sensitivity test: RNA / RNA:DNA vs DNA
#
# For each predictor, paired Wilcoxon signed-rank test across 21 ASVs
# Tests whether one data type is consistently more sensitive than DNA
#
# Output:
#   wilcoxon_results.csv          — full test results
#   fig_directional_sensitivity.png — publication figure (dot + CI plot)
# =============================================================================



DATA_PATH = '%sgam_results/02_sensitivity_per_triplet.csv' % config.data_directory
OUT_CSV   = "%sgam_results/wilcoxon_results.csv" % config.data_directory
OUT_FIG   = "%sgam_results/fig_directional_sensitivity.png" % config.data_directory



mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

PRED_ORDER = [
    "salinity", "specific_conductivity", "total_nitrogen", "ph",
    "doc", "secchi_depth", "dissolved_oxygen", "water_temp",
    "day_of_year", "days", "total_phosphorus",
]
PRED_LABELS = {
    "salinity":               "Salinity",
    "specific_conductivity":  "Sp. conductivity",
    "total_nitrogen":         "Total N",
    "ph":                     "pH",
    "doc":                    "DOC",
    "secchi_depth":           "Secchi depth",
    "dissolved_oxygen":       "Dissolved O₂",
    "water_temp":             "Water temp",
    "day_of_year":            "Day of year",
    "days":                   "Days",
    "total_phosphorus":       "Total P",
}
COLORS = {
    "RNA vs DNA":     "#0F6E56",   # green
    "RNA:DNA vs DNA": "#854F0B",   # amber
}

# =============================================================================
# 1. PAIRED WILCOXON TESTS
# =============================================================================
df = pd.read_csv(DATA_PATH)

results = []
for pred in PRED_ORDER:
    sub     = df[df["predictor"] == pred]
    dna     = sub[sub["dtype"] == "dna"    ].set_index("asv_id")["mean_abs_deriv"]
    rna     = sub[sub["dtype"] == "rna"    ].set_index("asv_id")["mean_abs_deriv"]
    rna_dna = sub[sub["dtype"] == "rna_dna"].set_index("asv_id")["mean_abs_deriv"]

    for pair_name, alt in [("RNA vs DNA", rna), ("RNA:DNA vs DNA", rna_dna)]:
        idx  = dna.index.intersection(alt.index)
        d    = alt[idx].values
        b    = dna[idx].values
        diff = d - b   # positive = alt more sensitive than DNA

        n = len(diff)
        n_greater = (diff > 0).sum()
        n_lesser  = (diff < 0).sum()

        stat, p  = stats.wilcoxon(d, b, alternative="two-sided",
                                  zero_method="wilcox")
        med_diff = np.median(diff)

        # Bootstrap 95% CI on the median difference
        rng = np.random.default_rng(42)
        boot_meds = np.array([
            np.median(rng.choice(diff, size=n, replace=True))
            for _ in range(5000)
        ])
        ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])

        # rank-biserial correlation as effect size (ranges -1 to +1)
        rb = 1 - (2 * stat) / (n * (n + 1) / 2)

        results.append({
            "predictor":       pred,
            "pred_label":      PRED_LABELS[pred],
            "comparison":      pair_name,
            "n_asv":           n,
            "n_alt_greater":   n_greater,
            "n_dna_greater":   n_lesser,
            "pct_alt_greater": round(100 * n_greater / n, 1),
            "median_diff":     round(med_diff, 6),
            "ci_lo":           round(ci_lo, 6),
            "ci_hi":           round(ci_hi, 6),
            "rank_biserial":   round(rb, 3),
            "W_stat":          stat,
            "p_raw":           p,
        })

res = pd.DataFrame(results)
_, p_adj, _, _ = multipletests(res["p_raw"], method="fdr_bh")
res["p_adj_BH"] = p_adj
res["sig"]      = p_adj < 0.05
res["direction"] = res.apply(
    lambda r: ("alt > DNA" if r["median_diff"] > 0 else "DNA > alt")
    if r["sig"] else "n.s.", axis=1
)

res.to_csv(OUT_CSV, index=False)
print("Results saved:", OUT_CSV)
print(f"\n{res['sig'].sum()} / {len(res)} tests significant (BH q < 0.05)")
print(res[res["sig"]][["pred_label","comparison","pct_alt_greater",
                        "median_diff","rank_biserial","p_adj_BH","direction"]]
      .to_string(index=False))

# =============================================================================
# 2. FIGURE — dot + CI plot with significance annotations
#
# For each predictor (y-axis), show two rows:
#   RNA vs DNA (green)       — top offset
#   RNA:DNA vs DNA (amber)   — bottom offset
#
# x-axis: median difference in mean |f'| (positive = alt more sensitive)
# Error bars: bootstrap 95% CI
# Filled dot = significant (BH q < 0.05); open dot = n.s.
# =============================================================================

pairs     = ["RNA vs DNA", "RNA:DNA vs DNA"]
n_pred    = len(PRED_ORDER)
offsets   = {"RNA vs DNA": 0.18, "RNA:DNA vs DNA": -0.18}

# Use within-predictor scaling so all predictors are visible together.
# Scale each predictor's differences to its own IQR of DNA sensitivity
# (so the x-axis is in units of "IQR of DNA sensitivity for that predictor").
dna_iqr = (
    df[df["dtype"] == "dna"]
    .groupby("predictor")["mean_abs_deriv"]
    .quantile(0.75)
    - df[df["dtype"] == "dna"]
    .groupby("predictor")["mean_abs_deriv"]
    .quantile(0.25)
).rename("iqr_dna").reset_index()

res = res.merge(dna_iqr, on="predictor", how="left")
# avoid division by zero for very flat predictors
res["iqr_dna"] = res["iqr_dna"].replace(0, np.nan)
res["scaled_diff"]  = res["median_diff"] / res["iqr_dna"]
res["scaled_ci_lo"] = res["ci_lo"]       / res["iqr_dna"]
res["scaled_ci_hi"] = res["ci_hi"]       / res["iqr_dna"]

fig, ax = plt.subplots(figsize=(8, 6))

ax.axvline(0, color="#aaa", lw=1, ls="--", zorder=0)

for pi, pred in enumerate(PRED_ORDER):
    for pair in pairs:
        row = res[(res["predictor"] == pred) & (res["comparison"] == pair)].iloc[0]
        y   = pi + offsets[pair]
        x   = row["scaled_diff"]
        lo  = row["scaled_ci_lo"]
        hi  = row["scaled_ci_hi"]
        col = COLORS[pair]
        sig = row["sig"]

        # CI bar
        ax.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=2, solid_capstyle="round")
        # cap ticks
        for xv in [lo, hi]:
            ax.plot([xv, xv], [y - 0.07, y + 0.07], color=col, lw=1.2, zorder=2)
        # point: filled if significant, open if not
        if sig:
            ax.scatter(x, y, color=col, s=55, zorder=4, edgecolors=col, linewidths=1)
        else:
            ax.scatter(x, y, color="white", s=55, zorder=4,
                       edgecolors=col, linewidths=1.2)

        # significance label
        if sig:
            q = row["p_adj_BH"]
            stars = "***" if q < 0.001 else ("**" if q < 0.01 else "*")
            ax.text(hi + 0.04, y, stars, va="center", ha="left",
                    fontsize=8, color=col, fontweight="bold")

ax.set_yticks(range(n_pred))
ax.set_yticklabels([PRED_LABELS[p] for p in PRED_ORDER], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(
    "Median difference in sensitivity (alt − DNA),\n"
    "scaled to within-predictor IQR of DNA",
    fontsize=9
)
ax.set_title(
    "Is RNA or RNA:DNA consistently more sensitive than DNA?\n"
    "Paired Wilcoxon test across 21 ASVs per predictor · BH-FDR corrected",
    pad=10
)
ax.margins(y=0.03)

# Legend
legend_elements = [
    mpatches.Patch(color=COLORS["RNA vs DNA"],
                   label="RNA vs DNA"),
    mpatches.Patch(color=COLORS["RNA:DNA vs DNA"],
                   label="RNA:DNA vs DNA"),
    plt.scatter([], [], color="grey",   s=45, label="Significant (q < 0.05)"),
    plt.scatter([], [], color="white",  s=45, edgecolors="grey",
                linewidths=1.2, label="Not significant"),
]
ax.legend(handles=legend_elements, loc="lower right",
          frameon=False, fontsize=8)

# Add n= annotation
ax.text(0.01, 1.01, "n = 21 ASVs per test",
        transform=ax.transAxes, fontsize=7.5,
        color="#666", ha="left", va="bottom")

fig.tight_layout()
fig.savefig(OUT_FIG)
plt.close(fig)
print("\nFigure saved:", OUT_FIG)