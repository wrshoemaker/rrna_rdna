import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
 
import config
import utils
 

data_path = '%sgam_results/02_sensitivity_per_triplet.csv' % config.data_directory

PRED_ORDER = ["salinity", "specific_conductivity", "total_nitrogen", "ph", "doc", "secchi_depth", "dissolved_oxygen", "water_temp", "total_phosphorus"]

PRED_LABELS = {
    "salinity":               "Salinity",
    "specific_conductivity":  "Sp. conductivity",
    "total_nitrogen":         "Total N",
    "ph":                     "pH",
    "doc":                    "DOC",
    "secchi_depth":           "Secchi depth",
    "dissolved_oxygen":       "Dissolved O\u2082",
    "water_temp":             "Water temp",
    "day_of_year":            "Day of year",
    "days":                   "Days",
    "total_phosphorus":       "Total P",
}
DTYPE_ORDER  = ["dna", "rna", "rna_dna"]
DTYPE_TITLES = {"dna": "rDNA", "rna": "rRNA", "rna_dna": "rRNA:rDNA"}
 



def make_per_asv_heatmap():
 
    df = pandas.read_csv(data_path)
 
    # Within-predictor z-score across ASVs for each dtype separately
    df["z"] = df.groupby(["predictor", "dtype"])["mean_abs_deriv"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1)
    )
 
    # Rank ASVs by mean z across ALL predictors and dtypes — most responsive top
    asv_order = (df.groupby("asv_id")["z"].mean().sort_values(ascending=False).index.tolist())
    n_asv  = len(asv_order)
    n_pred = len(PRED_ORDER)

    asv_order_formatted = [s.split('_')[0] + ' ' + s.split('_')[1].lstrip("0") for s in asv_order]
 
    # Diverging colormap: blue (below avg) → white → red (above avg)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bwr_mid", ["#185FA5", "#f5f4f0", "#A32D2D"]
    )
    zmax = numpy.nanpercentile(numpy.abs(df["z"]), 97)
    norm = mcolors.Normalize(vmin=-zmax, vmax=zmax)
 
    # Cell dimensions in inches
    cw, ch = 0.55, 0.38
 
    fig, axes = plt.subplots(
        1, 3,
        figsize=(3 * (n_pred * cw + 1.6), n_asv * ch + 1.4),
        gridspec_kw={"wspace": 0.08}
    )
 
    for col_idx, dtype in enumerate(DTYPE_ORDER):
        ax  = axes[col_idx]
        sub = df[df["dtype"] == dtype]
 
        pivot_z = (
            sub.pivot(index="asv_id", columns="predictor", values="z")
            .reindex(index=asv_order, columns=PRED_ORDER)
        )
        pivot_raw = (
            sub.pivot(index="asv_id", columns="predictor", values="mean_abs_deriv")
            .reindex(index=asv_order, columns=PRED_ORDER)
        )
 
        vals_z   = pivot_z.values.astype(float)
        vals_raw = pivot_raw.values.astype(float)
 
        for ri in range(n_asv):
            for ci in range(n_pred):
                z   = vals_z[ri, ci]
                raw = vals_raw[ri, ci]
                if numpy.isnan(z):
                    continue
 
                color = cmap(norm(z))
                rect  = plt.Rectangle(
                    [ci - 0.5, ri - 0.5], 1, 1,
                    facecolor=color, edgecolor="white", lw=0.3
                )
                ax.add_patch(rect)
 
                # Text colour by luminance
                r, g, b, _ = color
                lum = 0.299*r + 0.587*g + 0.114*b
                tc  = "white" if lum < 0.55 else "#333"
 
                # Format raw value compactly
                if raw >= 10:
                    label = f"{raw:.0f}"
                elif raw >= 1:
                    label = f"{raw:.2f}"
                elif raw >= 0.01:
                    label = f"{raw:.3f}"
                else:
                    # scientific notation e.g. 5.7e-3
                    s = f"{raw:.1e}"
                    s = s.replace("e-0", "e-").replace("e+0", "e")
                    label = s
 
                #ax.text(ci, ri, label,
                #        ha="center", va="center",
                #        fontsize=5.5, color=tc)
 
        ax.set_xlim(-0.5, n_pred - 0.5)
        ax.set_ylim(-0.5, n_asv  - 0.5)
        ax.invert_yaxis()
 
        # x-axis: predictor labels, rotated
        ax.set_xticks(range(n_pred))
        ax.set_xticklabels([PRED_LABELS[p] for p in PRED_ORDER], rotation=40, ha="right", fontsize=10)
        ax.tick_params(length=0)
 
        # y-axis: ASV labels on leftmost panel only
        ax.set_yticks(range(n_asv))
        if col_idx == 0:
            ax.set_yticklabels(asv_order_formatted, fontsize=7.5)
            ax.tick_params(axis="y", length=0, pad=3)
        else:
            ax.set_yticklabels([])
 
        ax.set_title(DTYPE_TITLES[dtype], fontsize=20, fontweight="bold", pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
 
    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Within-predictor z-score\n(mean |f\u2032|)", fontsize=18, labelpad=8)
    cb.ax.tick_params(labelsize=7.5)
 
    #fig.suptitle(
    #    "Per-ASV sensitivity to environmental predictors\n"
    #    "Colour = within-predictor z-score of mean |f\u2032|   \u00b7   "
    #    "Number = raw mean |f\u2032|   \u00b7   "
    #    "ASVs ranked by mean z-score (most responsive top)",
    #    fontsize=8.5, y=1.01
    #)
 
    fig_name = "%sfig_per_asv_sensitivity_heatmap.png" % config.analysis_directory
    fig.savefig(fig_name, format="png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close()
    print("Saved:", fig_name)



make_per_asv_heatmap()
