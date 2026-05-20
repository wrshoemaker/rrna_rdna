

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

n_boot = 1000
alpha = 0.05

sens_triplets_path = '%sgam_results/02_sensitivity_per_triplet.csv' % config.data_directory
#OUT_CSV   = "%sgam_results/wilcoxon_results.csv" % config.data_directory
#OUT_FIG   = "%sgam_results/fig_directional_sensitivity.png" % config.data_directory
curves_path = "%sgam_results/04_diff_smooth_curves.csv" % config.data_directory



pairs_colors_split = {'rRNA vs rDNA': (utils.dna_rna_color_dict['RNA'], utils.dna_rna_color_dict['DNA']), 'rRNA:rDNA vs rDNA': (utils.dna_rna_color_dict['ratio'], utils.dna_rna_color_dict['DNA'])}
pairs_colors = {'rRNA vs rDNA': utils.dna_rna_color_dict['RNA'], 'rRNA:rDNA vs rDNA': utils.dna_rna_color_dict['ratio']}

pair_rna = 'rRNA vs rDNA'
pair_rnadna = 'rRNA:rDNA vs rDNA'


pair_map = {
    pair_rnadna: "DNA vs RNA:DNA",
    pair_rna:    "DNA vs RNA",
}




df_sens_triplets = pandas.read_csv(sens_triplets_path)
results = []
for pred in utils.env_variable_to_plot:
    sub = df_sens_triplets[df_sens_triplets["predictor"] == pred]
    dna = sub[sub["dtype"]=="dna"].set_index("asv_id")["mean_abs_deriv"]
    rna = sub[sub["dtype"]=="rna"].set_index("asv_id")["mean_abs_deriv"]
    rna_dna = sub[sub["dtype"]=="rna_dna"].set_index("asv_id")["mean_abs_deriv"]

    for pair_name, alt in [("rRNA vs rDNA", rna), ("rRNA:rDNA vs rDNA", rna_dna)]:
        idx = dna.index.intersection(alt.index)
        d = alt[idx].values
        b = dna[idx].values
        # positive = alt more sensitive than DNA
        diff = d - b

        n = len(diff)
        n_greater = (diff > 0).sum()
        n_lesser = (diff < 0).sum()

        stat, p = stats.wilcoxon(d, b, alternative="two-sided", zero_method="wilcox")
        med_diff = numpy.median(diff)

        # Bootstrap 95% CI on the median difference
        rng = numpy.random.default_rng(42)
        boot_meds = numpy.array([numpy.median(rng.choice(diff, size=n, replace=True)) for _ in range(n_boot)])
        ci_lo, ci_hi = numpy.percentile(boot_meds, [2.5, 97.5])
        # rank-biserial correlation as effect size (ranges -1 to +1)
        rb = 1 - (2 * stat) / (n * (n + 1) / 2)

        results.append({
            "predictor":       pred,
            "pred_label":      utils.env_variable_no_unit_label_dict[pred],
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



res = pandas.DataFrame(results)
_, p_adj, _, _ = multipletests(res["p_raw"], method="fdr_bh")
res["p_adj_BH"] = p_adj
res["sig"]      = p_adj < alpha
res["direction"] = res.apply(lambda r: ("alt > DNA" if r["median_diff"] > 0 else "DNA > alt") if r["sig"] else "n.s.", axis=1)


pairs = ["rRNA vs rDNA", "rRNA:rDNA vs rDNA"]
n_pred = len(utils.env_variable_to_plot)
offsets = {"rRNA vs rDNA": 0.18, "rRNA:rDNA vs rDNA": -0.18}

dna_iqr = (df_sens_triplets[df_sens_triplets["dtype"] == "dna"].groupby("predictor")["mean_abs_deriv"].quantile(0.75) - df_sens_triplets[df_sens_triplets["dtype"] == "dna"].groupby("predictor")["mean_abs_deriv"].quantile(0.25)).rename("iqr_dna").reset_index()
res = res.merge(dna_iqr, on="predictor", how="left")
# avoid division by zero for very flat predictors
res["iqr_dna"] = res["iqr_dna"].replace(0, numpy.nan)
res["scaled_diff"]  = res["median_diff"] / res["iqr_dna"]
res["scaled_ci_lo"] = res["ci_lo"] / res["iqr_dna"]
res["scaled_ci_hi"] = res["ci_hi"] / res["iqr_dna"]


#######
# save for blomberg K
sig_preds_rna_dna = (res[(res["comparison"] == 'rRNA:rDNA vs rDNA') & (res["sig"]) & (res["median_diff"] > 0)]["predictor"].tolist())
pandas.DataFrame({"predictor": sig_preds_rna_dna}).to_csv("%sgam_results/sig_preds_rna_dna.csv" % config.data_directory,index=False)
#######


pred_rank = (res.groupby("predictor")["scaled_diff"].median().sort_values().index.tolist())
pred_rank = [p for p in pred_rank if p in utils.env_variable_to_plot]

#pred_rank = (res[res["comparison"] == 'rRNA:rDNA vs rDNA'].set_index("predictor")["scaled_diff"].sort_values().index.tolist())
#pred_rank = [p for p in pred_rank if p in utils.env_variable_to_plot]
n_pred = len(pred_rank)

pairs   = ["rRNA vs rDNA", "rRNA:rDNA vs rDNA"]
offsets = {"rRNA vs rDNA": 0.18, "rRNA:rDNA vs rDNA": -0.18}
pairs_colors = {"rRNA vs rDNA": utils.dna_rna_color_dict['RNA'], "rRNA:rDNA vs rDNA": utils.dna_rna_color_dict['ratio']}
pairs_colors_split = {"rRNA vs rDNA": (utils.dna_rna_color_dict['RNA'], utils.dna_rna_color_dict['DNA']), "rRNA:rDNA vs rDNA": (utils.dna_rna_color_dict['ratio'], utils.dna_rna_color_dict['DNA']),}


# alt > DNA direction
sig_preds_rna_dna = (res[(res["comparison"] == "rRNA:rDNA vs rDNA") & (res["sig"]) & (res["median_diff"] > 0)]["predictor"].tolist())






def make_sensitivity_plot(n_perm=9999):
   
    #fig = plt.figure(figsize=(12, 16), layout="constrained")

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.2, 1, 1, 1])
    outer = GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1], hspace=0.4)

    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[0.08, 0.92, 0])
    bottom = GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[1:], wspace=0.25, hspace=0.35)

    #ax_top = fig.add_subplot(gs[0, :])
    ax_top = fig.add_subplot(top[1:])
    ax_top.axvline(0, color='k', lw=2, ls=":", zorder=0)
    x_max = numpy.percentile(res["scaled_ci_hi"].dropna(), 90)
    x_min = res["scaled_ci_lo"].min()
    ax_top.set_xlim(x_min - 0.05, x_max + 0.1)
    #pad   = (x_max - x_min) * 0.04
    #ax_top.set_xlim(x_min - pad, x_max + pad)

    for pi, pred in enumerate(pred_rank[::-1]):
        for pair in pairs:
            row = res[(res["predictor"] == pred) & (res["comparison"] == pair)].iloc[0]
            y   = pi + offsets[pair]
            x   = row["scaled_diff"]
            lo  = row["scaled_ci_lo"]
            hi  = row["scaled_ci_hi"]
            col = pairs_colors[pair]
            sig = row["sig"]
            ax_top.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=2, solid_capstyle="round")
            for xv in [lo, hi]:
                ax_top.plot([xv, xv], [y - 0.07, y + 0.07], color=col, lw=1.2, zorder=2)
            fc = col if sig else "white"
            ax_top.scatter(x, y, color=fc, s=55, zorder=4, edgecolors=col, linewidths=1.2)

    ax_top.set_yticks(range(n_pred))
    #ax_top.set_yticklabels(
    #    [utils.env_variable_no_unit_label_dict[p] for p in pred_rank[::-1]],
    #    fontsize=12, rotation=30
    #)
    ax_top.set_yticklabels([utils.env_variable_no_unit_label_dict[p] for p in pred_rank[::-1]], fontsize=12, rotation=25)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Median scaled difference in sensitivity relative to rDNA", fontsize=14)

    legend_elements = [
        mpatches.Patch(color=pairs_colors[pair_rna],    label="rRNA vs. rDNA"),
        mpatches.Patch(color=pairs_colors[pair_rnadna], label="rRNA:DNA vs. rDNA"),
        plt.scatter([], [], color="grey",  s=45, label=r'$P < 0.05$'),
        plt.scatter([], [], color="white", s=45, edgecolors="grey",
                    linewidths=1.2, label=r'$P \nless 0.05$')]
    ax_top.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=8)

    # updated return values
    all_shape_df, all_shape_data = run_gradient_shape_test(n_perm=n_perm)
    # all 9 predictors in 3 rows
    env_variable_nested = [
        ['water_temp', 'dissolved_oxygen', 'secchi_depth' ],
        ['doc', 'ph', 'total_nitrogen'],
        ['salinity', 'specific_conductivity', 'total_phosphorus'],
    ]


    fig.text(0.06, 0.38, "Scaled sensitivity advantage over rDNA", va="center", ha="center", rotation="vertical", fontsize=20)
    
    for pred_chunk_idx, pred_chunk in enumerate(env_variable_nested):
        for pred_idx, pred in enumerate(pred_chunk):
            
            #ax_pred = fig.add_subplot(gs_bot[pred_chunk_idx, pred_idx])
            #ax_pred = fig.add_subplot(gs[pred_chunk_idx + 1, pred_idx])
            ax_pred = fig.add_subplot(bottom[pred_chunk_idx, pred_idx])

            bottom
            ax_pred.axhline(0, color="k", lw=2, ls=":", zorder=1)

            # 5. loop over both pairs
            for pair_name in [pair_rnadna, pair_rna]:
                
                col = pairs_colors[pair_name]

                # 4. updated key
                binned = all_shape_data[(pair_name, pred)]
                row_shp = all_shape_df[pair_name][all_shape_df[pair_name]["predictor"] == pred].iloc[0]

                x  = binned["x_mid"].values
                y  = binned["rna_dna_advantage"].values
                se = binned["se_adv"].values

                #ls = "-" if row_shp["sig"] else "--"
                ls = '-'
                fc = col if row_shp["sig"] else "white"

                ax_pred.fill_between(x, y - se, y + se, color=col, alpha=0.2, lw=0, zorder=2)
                ax_pred.plot(x, y, color=col, lw=2, ls=ls, zorder=3)
                ax_pred.scatter(x, y, color=fc, s=50, zorder=4, edgecolors=col, lw=0.5)

            #ax_pred.set_xlabel(utils.env_variable_label_dict[pred], fontsize=11)
            ax_pred.set_xlabel(utils.env_variable_label_dict[pred], fontsize=14, labelpad=2)
            ax_pred.tick_params(axis='x', pad=2)
            #if pred_idx == 0:
            #    ax_pred.set_ylabel("Sensitivity advantage\nover rDNA", fontsize=10)
            #else:
            #    ax_pred.set_ylabel("")
            ax_pred.tick_params(labelleft=True) 
            #ax_pred.spines["top"].set_visible(False)
            #ax_pred.spines["right"].set_visible(False)

    #fig.subplots_adjust(hspace=0.5, wspace=0.40)
    fig.tight_layout()
    fig_name = "%sgam_sensitivity.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches="tight", pad_inches=0.4, dpi=600)
    plt.close()

    print(all_shape_df)


def gradient_shape(curves, pair, pred, n_bins=10, n_perm=9999, seed=123456789):
    sub = curves[(curves["pair_label"] == pair) & (curves["predictor"] == pred)].copy()

    # Mean advantage curve across ASVs at each evaluation point
    mean_curve = (
        sub.groupby("x_value")
        .agg(advantage=("difference", lambda x: -x.mean()))
        .reset_index()
        .sort_values("x_value")
    )

    # Observed Spearman rho on mean curve
    r_obs, _ = stats.spearmanr(mean_curve["x_value"], mean_curve["advantage"])

    # Permutation null: shuffle advantage values relative to x positions
    # Shuffling ASV labels is incorrect — all ASVs share the same x_value grid
    # so the mean at each x is invariant to label permutation.
    # Instead permute the advantage values themselves, breaking the
    # x-advantage association while preserving their marginal distribution.
    rng      = numpy.random.default_rng(seed)
    adv_vals = mean_curve["advantage"].values
    x_vals   = mean_curve["x_value"].values
    null_rhos = numpy.empty(n_perm)

    for b in range(n_perm):
        perm_adv     = rng.permutation(adv_vals)
        null_rhos[b], _ = stats.spearmanr(x_vals, perm_adv)

    # Two-sided permutation p-value, floored at 1/n_perm
    p_perm = (numpy.abs(null_rhos) >= numpy.abs(r_obs)).mean()
    p_perm = max(p_perm, 1.0 / n_perm)

    # Binned summary for plotting only
    sub["bin_idx"] = pandas.qcut(
        sub["x_value"], q=n_bins, labels=False, duplicates="drop"
    )
    binned = (
        sub.groupby("bin_idx")
        .agg(
            x_mid     = ("x_value",    "median"),
            mean_diff = ("difference", "mean"),
            se_diff   = ("difference", lambda x: x.std() / numpy.sqrt(len(x))),
        )
        .reset_index()
    )
    binned["rna_dna_advantage"] = -binned["mean_diff"]
    binned["se_adv"]            =  binned["se_diff"]


    return binned, r_obs, p_perm



def run_gradient_shape_test_old():

    # Analysis 1: Gradient shape test
    shape_records = []
    shape_data    = {}
 
    df_curves = pandas.read_csv(curves_path)
 
    for pred in sig_preds_rna_dna:
        binned, r_sp, p_sp = gradient_shape(df_curves, "rDNA vs rRNA:DNA", pred, n_bins=10, n_perm=9999, seed=123456789)
        shape_data[pred] = binned
        shape_records.append({
            "predictor":  pred,
            "pred_label": utils.env_variable_no_unit_label_dict[pred],
            "spearman_r": r_sp,
            "spearman_p": p_sp,
            "pattern": (
                "monotone increasing" if (r_sp >  0.6 and p_sp < alpha) else
                "monotone decreasing" if (r_sp < -0.6 and p_sp < alpha) else
                "non-monotone"
            ),
            "peak_x_mid": round(binned.loc[binned["rna_dna_advantage"].idxmax(), "x_mid"], 3),
            "peak_adv":   round(binned["rna_dna_advantage"].max(), 4),
        })
 
    shape_df = pandas.DataFrame(shape_records)

    #shape_df.to_pickle("%sshape_df.pkl" % config.data_directory)
    # Summary by pattern type
    #monotone_inc = shape_df[shape_df["pattern"] == "monotone increasing"]["predictor"].tolist()
    #monotone_dec = shape_df[shape_df["pattern"] == "monotone decreasing"]["predictor"].tolist()
    #non_monotone = shape_df[shape_df["pattern"] == "non-monotone"]["predictor"].tolist()

    #print("\n=== Gradient shape results ===")
    #print(shape_df[["predictor","spearman_r","spearman_p","pattern","peak_x_mid","peak_adv"]]
    #    .round({"spearman_r":3,"spearman_p":4,"peak_x_mid":3,"peak_adv":4})
    #    .to_string(index=False))

    #print(f"\nMonotone increasing ({len(monotone_inc)}): {', '.join([utils.env_variable_no_unit_label_dict[p] for p in monotone_inc])}")
    #print(f"Monotone decreasing ({len(monotone_dec)}): {', '.join([utils.env_variable_no_unit_label_dict[p] for p in monotone_dec])}")
    #print(f"Non-monotone ({len(non_monotone)}): {', '.join([utils.env_variable_no_unit_label_dict[p] for p in non_monotone])}")

    #print(f"\nInterpretation: rRNA:rDNA advantage concentrates toward oligotrophic")
    #print(f"conditions for {len(monotone_inc)+len(monotone_dec)}/{len(shape_df)} significant predictors")
    #print(f"(high water temp/DO/Secchi, low DOC — Spearman |ρ| ≥ 0.745, p ≤ 0.013)")
    #print(f"pH and total N show no gradient concentration (p > 0.40) — likely taxon-specific effects")

        
    #shape_df.to_csv(f"{OUT_DIR}/gradient_shape_results.csv", index=False)
    #print(shape_df[["predictor","spearman_r","spearman_p","pattern","peak_x_mid"]].to_string(index=False))

    # monotone increasing/decreasing rRNA:rDNA sensitivity advantage at high/low values
    # rRNA:rDNA is most environmentally responsive relative to rDNA under oligotrophic conditions: warm, clear, well-oxygenated, low organic carbon
    # also, onogiotrophic/stratification as single environmental gradient
    # pH and nitrogen don't show on this axis ==> taxon-specific effects?

    return shape_df, shape_data






def run_gradient_shape_test(n_perm=9999):

    df_curves = pandas.read_csv(curves_path)


    for pair_label in ["DNA vs RNA", "DNA vs RNA:DNA"]:
        sub = df_curves[
            (df_curves["pair_label"] == pair_label) &
            (df_curves["predictor"]  == "water_temp")
        ]
        print(f"\n{pair_label}:")
        print(f"  mean difference: {sub['difference'].mean():.6f}")
        print(f"  min:             {sub['difference'].min():.6f}")
        print(f"  max:             {sub['difference'].max():.6f}")
        print(f"  first 3 values:  {sub['difference'].values[:3]}")
        
    # pair_name (matches res["comparison"]) -> curve label in df_curves["pair_label"]

    all_shape_data = {}   # key: (pair_name, pred)
    all_shape_df   = {}   # key: pair_name

    for pair_name, curve_label in pair_map.items():
        shape_records = []

        for pred in utils.env_variable_to_plot:
            binned, r_sp, p_sp = gradient_shape(
                df_curves, curve_label, pred,
                n_bins=10, n_perm=n_perm, seed=123456789
            )
            # gradient_shape computes advantage = -mean(difference)
            # For "DNA vs RNA:DNA": difference = DNA - RNA:DNA
            #   => advantage = RNA:DNA - DNA  (positive = RNA:DNA more sensitive)
            # For "DNA vs RNA":    difference = DNA - RNA
            #   => advantage = RNA - DNA      (positive = RNA more sensitive)
            # Same function works for both pairs — sign convention is consistent

            if len(binned) == 0:
                print(f"EMPTY: pair='{curve_label}'  pred='{pred}'")
                print(f"  pair_label values in curves: {df_curves['pair_label'].unique()}")
                print(f"  predictor values in curves:  {df_curves['predictor'].unique()}")
                continue

            all_shape_data[(pair_name, pred)] = binned

            shape_records.append({
                "predictor":  pred,
                "pred_label": utils.env_variable_no_unit_label_dict[pred],
                "comparison": pair_name,
                "spearman_r": r_sp,
                "spearman_p": p_sp,
                "pattern": (
                    "monotone increasing" if (r_sp >  0.6 and p_sp < alpha) else
                    "monotone decreasing" if (r_sp < -0.6 and p_sp < alpha) else
                    "non-monotone"
                ),
                "peak_x_mid": round(binned.loc[binned["rna_dna_advantage"].idxmax(), "x_mid"], 3),
                "peak_adv":   round(binned["rna_dna_advantage"].max(), 4),
            })

        shape_df = pandas.DataFrame(shape_records)

        # BH-FDR correction across all 9 predictors for this pair
        _, shape_df["p_adj_BH"], _, _ = multipletests(
            shape_df["spearman_p"], method="fdr_bh"
        )
        shape_df["sig"] = shape_df["p_adj_BH"] < alpha

        all_shape_df[pair_name] = shape_df

    return all_shape_df, all_shape_data








def run_sign_consistency_test():

    # Analysis 2: Sign consistency binomial test
    print("\nRunning sign consistency analysis...")
    sign_records = []
    for pair_name, alt_dtype in [("RNA vs DNA","rna"), ("RNA:DNA vs DNA","rna_dna")]:
        signs = []
        for pred in utils.env_variable_to_plot:
            sub = df_sens_triplets[df_sens_triplets["predictor"] == pred]
            dna = sub[sub["dtype"] == "dna"].set_index("asv_id")["mean_abs_deriv"]
            alt = sub[sub["dtype"] == alt_dtype].set_index("asv_id")["mean_abs_deriv"]
            idx = dna.index.intersection(alt.index)
            signs.append(numpy.median(alt[idx].values - dna[idx].values) > 0)

        n_pos = sum(signs)
        binom = stats.binomtest(n_pos, len(signs), p=0.5, alternative="two-sided")
        print(f"{pair_name}: {n_pos}/{len(utils.env_variable_to_plot)} predictors show alt > DNA, P = {binom.pvalue:.4f}")
        sign_records.append({
            "comparison": pair_name,
            "n_preds_alt_greater": n_pos,
            "n_preds_total": len(signs),
            "binomial_p": round(binom.pvalue, 5),
            "sig": binom.pvalue < 0.05,})

    sign_df = pandas.DataFrame(sign_records)

    #
    # print(res[(res["comparison"] == "rRNA:DNA vs rDNA") &(res["median_diff"] < 0)]["predictor"].values)
    # only tota_phosphorus is not significant
    # under high-P conditions (eutrophic) rRNA:rDNA advantage disappears



def run_breadth_test():

    # Analysis 3: Breadth of response (prop_sig_region)
    print("\nRunning breadth of response analysis...")
    breadth_records = []
    for pred in utils.env_variable_to_plot:
        sub = df_sens_triplets[df_sens_triplets["predictor"] == pred]
        dna = sub[sub["dtype"] == "dna"].set_index("asv_id")["prop_sig_region"]
        for pair_name, alt_dtype in [("RNA vs DNA","rna"),("RNA:DNA vs DNA","rna_dna")]:
            alt = sub[sub["dtype"] == alt_dtype].set_index("asv_id")["prop_sig_region"]
            idx = dna.index.intersection(alt.index)
            d, b = alt[idx].values, dna[idx].values
            diff = d - b
            n_nz = (diff != 0).sum()
            if n_nz < 3:
                p_raw = numpy.nan
            else:
                _, p_raw = stats.wilcoxon(d, b, alternative="two-sided", zero_method="wilcox")
            breadth_records.append({
                "predictor":     pred,
                "comparison":    pair_name,
                "median_diff":   round(numpy.median(diff), 4),
                "n_alt_greater": (diff > 0).sum(),
                "p_raw":         p_raw,
            })

    breadth_df = pandas.DataFrame(breadth_records)
    valid = breadth_df["p_raw"].notna()
    _, p_adj, _, _ = multipletests(breadth_df.loc[valid, "p_raw"], method="fdr_bh")
    breadth_df.loc[valid, "p_adj_BH"] = p_adj
    print(breadth_df)
    breadth_df["sig"] = breadth_df["p_adj_BH"] < 0.05
    sig_breadth = breadth_df[breadth_df["sig"]]
    print(f"Breadth significant (BH q<0.05): {len(sig_breadth)} results")
    #print(sig_breadth[["predictor","comparison","n_alt_greater","median_diff","p_adj_BH"]].to_string(index=False))
    #breadth_df.to_csv(f"{OUT_DIR}/sign_breadth_results.csv", index=False)

    # no significant results
    # rRNA:rDNA sensitivity advantage is about magnitude, not breadth



if __name__ == "__main__":

    print('GAM sensitivity analysis')

    #run_sign_consistency_test()
    #run_breadth_test()

    #run_gradient_shape_test()
    make_sensitivity_plot()

    #all_shape_df, all_shape_data = run_gradient_shape_test()

    #print(all_shape_data)

    #print(shape_df)

