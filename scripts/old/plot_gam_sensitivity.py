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
alpha  = 0.05

sens_triplets_path = '%sgam_results/02_sensitivity_per_triplet.csv' % config.data_directory
deriv_curves_path  = '%sgam_results/08_derivative_curves.csv'       % config.data_directory

pairs_colors = {
    'rRNA vs rDNA':      utils.dna_rna_color_dict['RNA'],
    'rRNA:rDNA vs rDNA': utils.dna_rna_color_dict['ratio'],
}
pairs_colors_split = {
    'rRNA vs rDNA':      (utils.dna_rna_color_dict['RNA'],   utils.dna_rna_color_dict['DNA']),
    'rRNA:rDNA vs rDNA': (utils.dna_rna_color_dict['ratio'], utils.dna_rna_color_dict['DNA']),
}

pair_rna    = 'rRNA vs rDNA'
pair_rnadna = 'rRNA:rDNA vs rDNA'

pair_dtype_map = {
    pair_rnadna: 'rna_dna',
    pair_rna:    'rna',
}


# =============================================================================
# WILCOXON — top panel
# =============================================================================

df_sens_triplets = pandas.read_csv(sens_triplets_path)
results = []
for pred in utils.env_variable_to_plot:
    sub = df_sens_triplets[df_sens_triplets['predictor'] == pred]
    dna     = sub[sub['dtype'] == 'dna'    ].set_index('asv_id')['mean_abs_deriv']
    rna     = sub[sub['dtype'] == 'rna'    ].set_index('asv_id')['mean_abs_deriv']
    rna_dna = sub[sub['dtype'] == 'rna_dna'].set_index('asv_id')['mean_abs_deriv']

    for pair_name, alt in [(pair_rna, rna), (pair_rnadna, rna_dna)]:
        idx  = dna.index.intersection(alt.index)
        d, b = alt[idx].values, dna[idx].values
        diff = d - b
        n    = len(diff)

        stat, p  = stats.wilcoxon(d, b, alternative='two-sided', zero_method='wilcox')
        med_diff = numpy.median(diff)

        rng = numpy.random.default_rng(42)
        boot_meds = numpy.array([
            numpy.median(rng.choice(diff, size=n, replace=True))
            for _ in range(n_boot)
        ])
        ci_lo, ci_hi = numpy.percentile(boot_meds, [2.5, 97.5])
        rb = 1 - (2 * stat) / (n * (n + 1) / 2)

        results.append({
            'predictor':       pred,
            'pred_label':      utils.env_variable_no_unit_label_dict[pred],
            'comparison':      pair_name,
            'n_asv':           n,
            'n_alt_greater':   (diff > 0).sum(),
            'n_dna_greater':   (diff < 0).sum(),
            'pct_alt_greater': round(100 * (diff > 0).mean(), 1),
            'median_diff':     round(med_diff, 6),
            'ci_lo':           round(ci_lo, 6),
            'ci_hi':           round(ci_hi, 6),
            'rank_biserial':   round(rb, 3),
            'W_stat':          stat,
            'p_raw':           p,
        })

res = pandas.DataFrame(results)
_, p_adj, _, _ = multipletests(res['p_raw'], method='fdr_bh')
res['p_adj_BH']  = p_adj
res['sig']       = p_adj < alpha
res['direction'] = res.apply(
    lambda r: ('alt > DNA' if r['median_diff'] > 0 else 'DNA > alt')
    if r['sig'] else 'n.s.', axis=1
)

dna_iqr = (
    df_sens_triplets[df_sens_triplets['dtype'] == 'dna']
    .groupby('predictor')['mean_abs_deriv']
    .quantile(0.75)
    - df_sens_triplets[df_sens_triplets['dtype'] == 'dna']
    .groupby('predictor')['mean_abs_deriv']
    .quantile(0.25)
).rename('iqr_dna').reset_index()

res = res.merge(dna_iqr, on='predictor', how='left')
res['iqr_dna'] = res['iqr_dna'].replace(0, numpy.nan)
res['scaled_diff'] = res['median_diff'] / res['iqr_dna']
res['scaled_ci_lo'] = res['ci_lo'] / res['iqr_dna']
res['scaled_ci_hi'] = res['ci_hi'] / res['iqr_dna']

# Save significant predictors for Blomberg K
sig_preds_rna_dna = (res[(res['comparison'] == pair_rnadna) & res['sig'] & (res['median_diff'] > 0)]['predictor'].tolist())
pandas.DataFrame({'predictor': sig_preds_rna_dna}).to_csv('%sgam_results/sig_preds_rna_dna.csv' % config.data_directory, index=False)

# Rank predictors by rRNA:rDNA scaled_diff
pred_rank = (
    res[res['comparison'] == pair_rnadna]
    .set_index('predictor')['scaled_diff']
    .sort_values()
    .index.tolist()
)
pred_rank = [p for p in pred_rank if p in utils.env_variable_to_plot]
n_pred    = len(pred_rank)

pairs   = [pair_rna, pair_rnadna]
offsets = {pair_rna: 0.18, pair_rnadna: -0.18}


# =============================================================================
# GRADIENT SHAPE — Option B (derivative differences)
# =============================================================================

def gradient_shape(df_deriv, pair_name, pred, n_bins=10, n_perm=9999, seed=123456789):
    alt_dtype = pair_dtype_map[pair_name]
    sub = df_deriv[df_deriv['predictor'] == pred].copy()

    dna_d = (
        sub[sub['dtype'] == 'dna']
        [['asv_id', 'x_value', 'derivative']]
        .rename(columns={'derivative': 'deriv_dna'})
    )
    alt_d = (
        sub[sub['dtype'] == alt_dtype]
        [['asv_id', 'x_value', 'derivative']]
        .rename(columns={'derivative': 'deriv_alt'})
    )

    merged = pandas.merge(dna_d, alt_d, on=['asv_id', 'x_value'])
    merged['delta'] = merged['deriv_alt'] - merged['deriv_dna']

    mean_curve = (
        merged.groupby('x_value')
        .agg(advantage=('delta', 'mean'))
        .reset_index()
        .sort_values('x_value')
    )

    r_obs, _ = stats.spearmanr(mean_curve['x_value'], mean_curve['advantage'])

    rng       = numpy.random.default_rng(seed)
    adv_vals  = mean_curve['advantage'].values
    x_vals    = mean_curve['x_value'].values
    null_rhos = numpy.empty(n_perm)

    for b in range(n_perm):
        perm_adv        = rng.permutation(adv_vals)
        null_rhos[b], _ = stats.spearmanr(x_vals, perm_adv)

    p_perm = (numpy.abs(null_rhos) >= numpy.abs(r_obs)).mean()
    p_perm = max(p_perm, 1.0 / n_perm)

    merged['bin_idx'] = pandas.qcut(
        merged['x_value'], q=n_bins, labels=False, duplicates='drop'
    )

    def bin_stats(g):
        asv_means = g.groupby('asv_id')['delta'].mean()
        n_asv     = len(asv_means)
        return pandas.Series({
            'x_mid':             g['x_value'].median(),
            'rna_dna_advantage': asv_means.mean(),
            'se_adv':            asv_means.std(ddof=1) / numpy.sqrt(n_asv),
            'n_asv':             n_asv,
        })

    binned = (
        merged.groupby('bin_idx')
        .apply(bin_stats, include_groups=False)
        .reset_index(drop=True)
    )

    return binned, r_obs, p_perm


def run_gradient_shape_test(n_perm=9999):

    df_deriv = pandas.read_csv(deriv_curves_path)
    print("Derivative curves dtypes:", df_deriv['dtype'].unique())
    print("Derivative curves predictors:", df_deriv['predictor'].unique())
    print("Rows:", len(df_deriv))

    all_shape_data = {}
    all_shape_df   = {}

    for pair_name in [pair_rnadna, pair_rna]:
        shape_records = []

        for pred in utils.env_variable_to_plot:
            binned, r_sp, p_sp = gradient_shape(
                df_deriv, pair_name, pred,
                n_bins=10, n_perm=n_perm, seed=123456789
            )

            if len(binned) == 0:
                print(f'EMPTY: pair={pair_name}  pred={pred}')
                continue

            all_shape_data[(pair_name, pred)] = binned

            shape_records.append({
                'predictor':  pred,
                'pred_label': utils.env_variable_no_unit_label_dict[pred],
                'comparison': pair_name,
                'spearman_r': r_sp,
                'spearman_p': p_sp,
                'pattern': (
                    'monotone increasing' if (r_sp >  0.6 and p_sp < alpha) else
                    'monotone decreasing' if (r_sp < -0.6 and p_sp < alpha) else
                    'non-monotone'
                ),
                'peak_x_mid': round(binned.loc[binned['rna_dna_advantage'].idxmax(), 'x_mid'], 3),
                'peak_adv':   round(binned['rna_dna_advantage'].max(), 4),
            })

        shape_df = pandas.DataFrame(shape_records)
        _, shape_df['p_adj_BH'], _, _ = multipletests(
            shape_df['spearman_p'], method='fdr_bh'
        )
        shape_df['sig'] = shape_df['p_adj_BH'] < alpha
        all_shape_df[pair_name] = shape_df

    return all_shape_df, all_shape_data


# =============================================================================
# MAIN FIGURE
# =============================================================================

def make_sensitivity_plot(n_perm=9999):

    fig    = plt.figure(figsize=(12, 12))
    outer  = GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1], hspace=0.4)
    top    = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                     width_ratios=[0.08, 0.92, 0])
    bottom = GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[1:],
                                     wspace=0.25, hspace=0.35)

    # ── top panel: median scaled difference ───────────────────────────────────
    ax_top = fig.add_subplot(top[1:])
    ax_top.axvline(0, color='k', lw=2, ls=':', zorder=0)
    x_max = numpy.percentile(res['scaled_ci_hi'].dropna(), 90)
    x_min = res['scaled_ci_lo'].min()
    ax_top.set_xlim(x_min - 0.05, x_max + 0.1)

    for pi, pred in enumerate(pred_rank[::-1]):
        for pair in pairs:
            row = res[(res['predictor'] == pred) & (res['comparison'] == pair)].iloc[0]
            y   = pi + offsets[pair]
            x   = row['scaled_diff']
            lo  = row['scaled_ci_lo']
            hi  = row['scaled_ci_hi']
            col = pairs_colors[pair]
            sig = row['sig']

            ax_top.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=2,
                        solid_capstyle='round')
            for xv in [lo, hi]:
                ax_top.plot([xv, xv], [y - 0.07, y + 0.07], color=col, lw=1.2, zorder=2)
            fc = col if sig else 'white'
            ax_top.scatter(x, y, color=fc, s=55, zorder=4, edgecolors=col, linewidths=1.2)

    # ── y-axis: colored ticks + bracket per predictor ─────────────────────────
    #
    # y-axis: two colored tick marks per predictor (no vertical bar)
    # label centred between them, separator lines between predictor groups
    ax_top.set_yticks([])
    ax_top.tick_params(axis='y', length=0)
    ax_top.invert_yaxis()

    xlim     = ax_top.get_xlim()
    xrange   = xlim[1] - xlim[0]
    tick_len = 0.018 * xrange
    tick_end = xlim[0] - tick_len            # left end of colored ticks
    label_x  = tick_end - 0.015 * xrange    # label x position

    for pi, pred in enumerate(pred_rank[::-1]):
        y_rna    = pi + offsets[pair_rna]
        y_rnadna = pi + offsets[pair_rnadna]
        y_mid    = pi   # exact midpoint between the two ticks

        # Colored tick for rRNA
        ax_top.plot([xlim[0], tick_end], [y_rna, y_rna],
                    color=pairs_colors[pair_rna], lw=2,
                    clip_on=False, zorder=5, solid_capstyle='butt')

        # Colored tick for rRNA:rDNA
        ax_top.plot([xlim[0], tick_end], [y_rnadna, y_rnadna],
                    color=pairs_colors[pair_rnadna], lw=2,
                    clip_on=False, zorder=5, solid_capstyle='butt')

        # Label centred between the two ticks.
        # rotation_mode='anchor' rotates around the ha/va anchor point
        # so the vertical centre stays locked to y_mid regardless of rotation.
        ax_top.text(label_x, y_mid,
                    utils.env_variable_no_unit_label_dict[pred],
                    ha='right', va='center', fontsize=11,
                    rotation=25, rotation_mode='anchor',
                    clip_on=False)

    # Dotted separator lines between each pair of environmental variables
    for pi in range(n_pred - 1):
        ax_top.axhline(pi + 0.5, color='k', lw=1, ls=':', zorder=1, alpha=1)

    ax_top.set_xlabel('Median scaled difference in sensitivity relative to rDNA', fontsize=14)

    legend_elements = [
        mpatches.Patch(color=pairs_colors[pair_rna],    label='rRNA'),
        mpatches.Patch(color=pairs_colors[pair_rnadna], label='rRNA:rDNA'),
        plt.scatter([], [], color='grey',  s=45, label=r'$P < 0.05$ (BH-FDR)'),
        plt.scatter([], [], color='white', s=45, edgecolors='grey', linewidths=1.2, label=r'$P \nless 0.05$')
    ]

    # mpatches.Patch(color='grey', alpha=0.15, label=r'$\bar{\delta}_j(x)$: $\pm$1 SE'),
    ax_top.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=8)

    # gradient shape panels
    all_shape_df, all_shape_data = run_gradient_shape_test(n_perm=n_perm)

    env_variable_nested = [
        ['water_temp', 'dissolved_oxygen', 'secchi_depth'],
        ['doc',        'ph',               'total_nitrogen'],
        ['salinity',   'specific_conductivity', 'total_phosphorus'],
    ]

    fig.text(0.06, 0.38,
             'Mean derivative difference\nrelative to rDNA',
             va='center', ha='center', rotation='vertical', fontsize=16)

    for pred_chunk_idx, pred_chunk in enumerate(env_variable_nested):
        for pred_idx, pred in enumerate(pred_chunk):

            ax_pred = fig.add_subplot(bottom[pred_chunk_idx, pred_idx])
            ax_pred.axhline(0, color='k', lw=2, ls=':', zorder=1)

            for pair_name in [pair_rnadna, pair_rna]:
                col    = pairs_colors[pair_name]
                binned = all_shape_data[(pair_name, pred)]

                x  = binned['x_mid'].values
                y  = binned['rna_dna_advantage'].values
                se = binned['se_adv'].values

                ax_pred.fill_between(x, y - se, y + se,
                                     color=col, alpha=0.15, lw=0, zorder=2)
                ax_pred.plot(x, y, color=col, lw=2, zorder=3)
                ax_pred.scatter(x, y, color=col, s=50, zorder=4,
                                edgecolors='white', lw=0.5)

            # Spearman rho annotation per pair
            y_pos = 0.97
            for pair_name in [pair_rnadna, pair_rna]:
                col     = pairs_colors[pair_name]
                row_shp = all_shape_df[pair_name][
                    all_shape_df[pair_name]['predictor'] == pred
                ].iloc[0]
                r     = row_shp['spearman_r']
                q     = row_shp['p_adj_BH']
                stars = '***' if q < 0.001 else ('**' if q < 0.01 else
                        ('*' if q < 0.05 else ''))
                ax_pred.text(0.97, y_pos, f'ρ={r:.2f}{stars}',
                             transform=ax_pred.transAxes,
                             ha='right', va='top', fontsize=7, color=col,
                             bbox=dict(facecolor='white', edgecolor='none',
                                       alpha=0.6, pad=1))
                y_pos -= 0.14

            ax_pred.set_xlabel(utils.env_variable_label_dict[pred],
                               fontsize=14, labelpad=2)
            ax_pred.tick_params(axis='x', pad=2)
            ax_pred.tick_params(labelleft=True)

    fig.tight_layout()
    fig_name = '%sgam_sensitivity.png' % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches='tight', pad_inches=0.4, dpi=600)
    plt.close()
    print(all_shape_df)


# =============================================================================
# SUPPORTING ANALYSES
# =============================================================================

def run_sign_consistency_test():
    print('\nRunning sign consistency analysis...')
    sign_records = []
    for pair_name, alt_dtype in [('RNA vs DNA','rna'), ('RNA:DNA vs DNA','rna_dna')]:
        signs = []
        for pred in utils.env_variable_to_plot:
            sub = df_sens_triplets[df_sens_triplets['predictor'] == pred]
            dna = sub[sub['dtype'] == 'dna'    ].set_index('asv_id')['mean_abs_deriv']
            alt = sub[sub['dtype'] == alt_dtype].set_index('asv_id')['mean_abs_deriv']
            idx = dna.index.intersection(alt.index)
            signs.append(numpy.median(alt[idx].values - dna[idx].values) > 0)

        n_pos = sum(signs)
        binom = stats.binomtest(n_pos, len(signs), p=0.5, alternative='two-sided')
        print(f'{pair_name}: {n_pos}/{len(utils.env_variable_to_plot)} predictors show alt > DNA, '
              f'P = {binom.pvalue:.4f}')
        sign_records.append({
            'comparison':          pair_name,
            'n_preds_alt_greater': n_pos,
            'n_preds_total':       len(signs),
            'binomial_p':          round(binom.pvalue, 5),
            'sig':                 binom.pvalue < 0.05,
        })
    return pandas.DataFrame(sign_records)


def run_breadth_test():
    print('\nRunning breadth of response analysis...')
    breadth_records = []
    for pred in utils.env_variable_to_plot:
        sub = df_sens_triplets[df_sens_triplets['predictor'] == pred]
        dna = sub[sub['dtype'] == 'dna'].set_index('asv_id')['prop_sig_region']
        for pair_name, alt_dtype in [('RNA vs DNA','rna'), ('RNA:DNA vs DNA','rna_dna')]:
            alt  = sub[sub['dtype'] == alt_dtype].set_index('asv_id')['prop_sig_region']
            idx  = dna.index.intersection(alt.index)
            d, b = alt[idx].values, dna[idx].values
            diff = d - b
            n_nz = (diff != 0).sum()
            p_raw = numpy.nan
            if n_nz >= 3:
                _, p_raw = stats.wilcoxon(d, b, alternative='two-sided',
                                          zero_method='wilcox')
            breadth_records.append({
                'predictor':     pred,
                'comparison':    pair_name,
                'median_diff':   round(numpy.median(diff), 4),
                'n_alt_greater': (diff > 0).sum(),
                'p_raw':         p_raw,
            })

    breadth_df = pandas.DataFrame(breadth_records)
    valid = breadth_df['p_raw'].notna()
    _, p_adj, _, _ = multipletests(breadth_df.loc[valid, 'p_raw'], method='fdr_bh')
    breadth_df.loc[valid, 'p_adj_BH'] = p_adj
    breadth_df['sig'] = breadth_df['p_adj_BH'] < 0.05
    print(breadth_df)
    print(f"Breadth significant (BH q<0.05): {breadth_df['sig'].sum()} results")
    return breadth_df


if __name__ == '__main__':

    print('GAM sensitivity analysis')

    #run_sign_consistency_test()
    #run_breadth_test()
    #run_gradient_shape_test()
    make_sensitivity_plot()
