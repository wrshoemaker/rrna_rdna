import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import warnings
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config
import utils


taxonomy_dict = utils.build_taxonomy_dict()


data_path = '%sgam_results/02_sensitivity_per_triplet.csv' % config.data_directory

PRED_ORDER = [
    "salinity", "specific_conductivity", "total_nitrogen", "ph", "doc",
    "secchi_depth", "dissolved_oxygen", "water_temp", "total_phosphorus"]

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
DTYPE_LABELS = {"dna": "rDNA", "rna": "rRNA", "rna_dna": "rRNA:rDNA"}

# -----------------------------------------------------------------------------
# 1. MAP asv_id → TROPHIC STATUS
# -----------------------------------------------------------------------------

lookup  = pandas.read_csv('%sgam_results/asv_id_lookup.csv' % config.data_directory)
asv_map = lookup[["asv_id", "base_name"]].drop_duplicates().copy()
asv_map["sequence"] = asv_map["base_name"].str.replace("^ASV_", "", regex=True)

def get_trophic(sequence):
    try:
        family = taxonomy_dict[sequence]["family"]
        return utils.family_trophic_status.get(family, "unknown")
    except KeyError:
        return "unknown"

asv_map["trophic"] = asv_map["sequence"].apply(get_trophic)
asv_trophic        = asv_map.set_index("asv_id")["trophic"].to_dict()

print("Trophic status counts:")
print(asv_map["trophic"].value_counts())

photo_asvs  = asv_map[asv_map["trophic"] == "phototroph"]["asv_id"].tolist()
hetero_asvs = asv_map[asv_map["trophic"] == "heterotroph"]["asv_id"].tolist()
print(f"\nPhototrophs  (n={len(photo_asvs)}):  {photo_asvs}")
print(f"Heterotrophs (n={len(hetero_asvs)}): {hetero_asvs}")

# -----------------------------------------------------------------------------
# 2. LOAD DATA & ADD TROPHIC STATUS
# -----------------------------------------------------------------------------

df = pandas.read_csv(data_path)
df["trophic"] = df["asv_id"].map(asv_trophic)

unmapped = df[df["trophic"].isna()]["asv_id"].unique()
if len(unmapped) > 0:
    print("WARNING — unmapped ASVs:", unmapped)

# -----------------------------------------------------------------------------
# 3. Z-SCORE ALL VALUES RELATIVE TO HETEROTROPH DISTRIBUTION
#    per predictor × dtype
#
#    z_i = (S_i - mean(S_hetero)) / std(S_hetero)
#
#    Heterotrophs will have a distribution centred near 0 with std ~1
#    Phototrophs will have z-scores reflecting their position in that distribution
# -----------------------------------------------------------------------------

def compute_zscores(df):
    """
    For each predictor × dtype, z-score all ASVs (hetero + photo)
    relative to the heterotroph mean and std.
    """
    records = []
    for pred in PRED_ORDER:
        for dtype in DTYPE_ORDER:
            sub    = df[(df["predictor"] == pred) & (df["dtype"] == dtype)]
            hetero = sub[sub["trophic"] == "heterotroph"]["mean_abs_deriv"]

            if len(hetero) < 3:
                continue

            h_mean = hetero.mean()
            h_std  = hetero.std(ddof=1)

            # Empirical 95% prediction interval from heterotroph distribution
            pi_lo, pi_hi = numpy.percentile(hetero.values, [2.5, 97.5])
            pi_lo_z      = (pi_lo - h_mean) / h_std if h_std > 0 else -2.0
            pi_hi_z      = (pi_hi - h_mean) / h_std if h_std > 0 else  2.0

            for _, row in sub.iterrows():
                z = (row["mean_abs_deriv"] - h_mean) / h_std if h_std > 0 else numpy.nan
                records.append({
                    "predictor":   pred,
                    "dtype":       dtype,
                    "asv_id":      row["asv_id"],
                    "trophic":     row["trophic"],
                    "z_score":     z,
                    "pi_lo_z":     pi_lo_z,
                    "pi_hi_z":     pi_hi_z,
                })

    return pandas.DataFrame(records)

zdf = compute_zscores(df)

# Save z-scores for phototrophs
photo_z = zdf[zdf["trophic"] == "phototroph"].copy()
photo_z["pred_label"]  = photo_z["predictor"].map(PRED_LABELS)
photo_z["dtype_label"] = photo_z["dtype"].map(DTYPE_LABELS)
print("\n=== Phototroph z-scores ===")
print(photo_z[["pred_label","dtype_label","asv_id","z_score"]]
      .sort_values("z_score", ascending=False)
      .to_string(index=False))
photo_z.to_csv(
    '%sgam_results/phototroph_zscores.csv' % config.data_directory,
    index=False
)

# -----------------------------------------------------------------------------
# 4. FIGURE
# -----------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

n_pred  = len(PRED_ORDER)
n_dtype = len(DTYPE_ORDER)

photo_colors = {
    asv: c
    for asv, c in zip(photo_asvs, plt.cm.tab10.colors[:len(photo_asvs)])
}

fig, axes = plt.subplots(
    n_dtype, n_pred,
    figsize=(n_pred * 1.1, n_dtype * 2.2),
    gridspec_kw={"hspace": 0.5, "wspace": 0.35}
)

rng_jitter = numpy.random.default_rng(42)

for di, dtype in enumerate(DTYPE_ORDER):
    for pi, pred in enumerate(PRED_ORDER):
        ax  = axes[di, pi]
        sub = zdf[(zdf["predictor"] == pred) & (zdf["dtype"] == dtype)]

        hetero_z  = sub[sub["trophic"] == "heterotroph"]["z_score"].values
        photo_sub = sub[sub["trophic"] == "phototroph"]

        if len(hetero_z) == 0:
            ax.set_visible(False)
            continue

        # Get prediction interval (same for all rows of this pred × dtype)
        pi_lo_z = sub["pi_lo_z"].iloc[0]
        pi_hi_z = sub["pi_hi_z"].iloc[0]

        # ── empirical 95% prediction interval shaded band ────────────────────
        ax.axhspan(pi_lo_z, pi_hi_z,
                   color="grey", alpha=0.12, zorder=1,
                   label="Heterotroph 95% PI")

        # ── reference lines ───────────────────────────────────────────────────
        ax.axhline(0,    color="black", lw=1,   ls=":",  zorder=2)  # mean
        ax.axhline( 1.96, color="grey", lw=0.7, ls="--", zorder=2)  # ~97.5th
        ax.axhline(-1.96, color="grey", lw=0.7, ls="--", zorder=2)  # ~2.5th

        # ── jittered heterotroph z-scores ────────────────────────────────────
        jitter = rng_jitter.uniform(-0.15, 0.15, size=len(hetero_z))
        ax.scatter(
            numpy.ones(len(hetero_z)) + jitter,
            hetero_z,
            color="grey", alpha=0.45, s=16, zorder=3, lw=0
        )

        # ── phototroph z-scores ───────────────────────────────────────────────
        for _, row in photo_sub.iterrows():
            col = photo_colors.get(row["asv_id"], "red")
            z   = row["z_score"]

            if numpy.isnan(z):
                continue

            ax.scatter(
                1, z,
                color=col, s=48, zorder=5,
                edgecolors="white", lw=0.5
            )

        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=6)

        # predictor label — top row only
        if di == 0:
            ax.set_title(PRED_LABELS[pred], fontsize=7, pad=3,
                         rotation=40, ha="left", rotation_mode="anchor")

        # dtype label — leftmost column only
        if pi == 0:
            ax.set_ylabel(DTYPE_LABELS[dtype], fontsize=8, labelpad=4)

# ── shared y-axis label ───────────────────────────────────────────────────────
fig.text(0.01, 0.5,
         "Sensitivity z-score\n(relative to heterotroph distribution)",
         va="center", ha="center", rotation="vertical", fontsize=9)

# ── shared legend ─────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color="grey", alpha=0.4,
                   label="Heterotroph 95% prediction interval"),
    plt.scatter([], [], color="grey", alpha=0.5, s=16,
                label="Heterotroph ASVs"),
] + [
    plt.scatter([], [], color=photo_colors[asv], s=48,
                label=asv)
    for asv in photo_asvs
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=len(photo_asvs) + 2,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02)
)

fig.suptitle(
    "Sensitivity z-scores: phototroph vs heterotroph ASVs\n"
    "z = (S\u1d62 \u2212 mean(S\u2095\u2091\u209c\u2091\u2090\u2092)) / std(S\u2095\u2091\u209c\u2091\u2090\u2092)   \u00b7   "
    "Shaded band = empirical 95% prediction interval   \u00b7   "
    "Dashed = z = \u00b11.96",
    fontsize=8.5, y=1.02
)

fig_name = "%sfig_trophic_sensitivity.png" % config.analysis_directory
fig.savefig(fig_name, format="png", bbox_inches="tight", pad_inches=0.3, dpi=300)
plt.close()
print(f"\nSaved: {fig_name}")
print("z-scores saved: gam_results/phototroph_zscores.csv")