import config
import numpy
import pandas
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from scipy import stats, signal
import pickle
import sine_parameter_utils
from statsmodels.stats.multitest import multipletests



env_variable_all = ['doc', 'secchi_depth', 'ph', 'dissolved_oxygen', 'water_temp', 'salinity', 'total_nitrogen', 'specific_conductivity', 'total_phosphorus']

n_boot = 1000
alpha = 0.05

sens_triplets_path = '%sgam_results/02_sensitivity_per_triplet.csv' % config.data_directory
deriv_curves_path = '%sgam_results/08_derivative_curves.csv' % config.data_directory

pairs_colors = {'rRNA vs rDNA': utils.dna_rna_color_dict['RNA'], 'rRNA:rDNA vs rDNA': utils.dna_rna_color_dict['ratio']}
pairs_colors_split = {'rRNA vs rDNA': (utils.dna_rna_color_dict['RNA'],   utils.dna_rna_color_dict['DNA']), 'rRNA:rDNA vs rDNA': (utils.dna_rna_color_dict['ratio'], utils.dna_rna_color_dict['DNA'])}
pair_rna = 'rRNA vs rDNA'
pair_rnadna = 'rRNA:rDNA vs rDNA'
pair_dtype_map = {pair_rnadna: 'rna_dna', pair_rna: 'rna'}


#-----

metadata_dict = utils.build_metadata_dict()
s_by_s, otu_labels, samples = utils.load_count_data()
param_env_dict = sine_parameter_utils.load_param_env_dict()    

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])
day_of_year = numpy.asarray([metadata_dict[s]['day_of_year'] for s in samples[(sample_type=='RNA')]])

minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


fig = plt.figure(figsize=(12, 8))
# Main layout: left half and right half
gs_main = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.8, 1], wspace=0.25)

# Left: 3x3 grid
gs_left = gs_main[0, 0].subgridspec(3, 3, wspace=0.3, hspace=0.2)
ax_left = [fig.add_subplot(gs_left[i, j]) for i in range(3) for j in range(3)]

# Right: 2 rows
gs_right = gs_main[0, 1].subgridspec(2, 1, hspace=0.13)

ax_r1 = fig.add_subplot(gs_right[0, 0])
ax_r2 = fig.add_subplot(gs_right[1, 0])


for env_variable_idx, env_variable in enumerate(env_variable_all):

    ax = ax_left[env_variable_idx]
    env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])
    # remove nans
    env_to_keep_idx = (~numpy.isnan(env_variable_array))
    env_variable_array_clean = env_variable_array[env_to_keep_idx]
    days_clean = days[env_to_keep_idx]
    env_variable_dict_idx = param_env_dict['env_variables_labels'].index(env_variable)
    
    ax.scatter(days_clean, env_variable_array_clean, s=8, alpha=1, zorder=2, c='k')
    days_range = numpy.linspace(min(days_clean), max(days_clean), 1000)
    sine_prediction = param_env_dict['amp_leastsq'][env_variable_dict_idx] * numpy.sin(param_env_dict['freq_leastsq'][env_variable_dict_idx] * days_range + param_env_dict['phase_leastsq'][env_variable_dict_idx]) + param_env_dict['param_mean_leastsq'][env_variable_dict_idx]
    ax.plot(days_range, sine_prediction, lw=2, ls='-', alpha=0.9, c='k', zorder=1, label='Sine function')
    ax.set_ylabel(utils.env_variable_label_dict[env_variable], fontsize=10)
    ax.set_xlim([0, max(days_clean)])
    ax.set_xticks(minor_days, minor=True)
    ax.set_xticks(major_days, minor=False)
    
    #if env_variable == len(env_variable)-1:
    ax.set_xticklabels(major_labels, minor=False, fontsize=6)
    if env_variable_idx > 5:
        ax.set_xlabel('Time (days)', fontsize=11)

    ax.xaxis.set_tick_params(labelsize=6)
    ax.yaxis.set_tick_params(labelsize=4)



# blank ticks
ax_r1.set_xticks([]) 
ax_r1.set_yticks([])

ax_r1.set_xlabel('Predicted\nCLR-transformed abundance', fontsize=11)
ax_r1.set_ylabel('Environmental variable', fontsize=11)





# now, barplot
df_sens_triplets = pandas.read_csv(sens_triplets_path)
results = []
for pred in utils.env_variable_to_plot:
    sub = df_sens_triplets[df_sens_triplets['predictor'] == pred]
    dna = sub[sub['dtype'] == 'dna'].set_index('asv_id')['mean_abs_deriv']
    rna = sub[sub['dtype'] == 'rna'].set_index('asv_id')['mean_abs_deriv']
    rna_dna = sub[sub['dtype'] == 'rna_dna'].set_index('asv_id')['mean_abs_deriv']

    for pair_name, alt in [(pair_rna, rna), (pair_rnadna, rna_dna)]:
        idx = dna.index.intersection(alt.index)
        d, b = alt[idx].values, dna[idx].values
        diff = d - b
        n = len(diff)

        stat, p = stats.wilcoxon(d, b, alternative='two-sided', zero_method='wilcox')
        med_diff = numpy.median(diff)

        rng = numpy.random.default_rng(42)
        boot_meds = numpy.array([ numpy.median(rng.choice(diff, size=n, replace=True)) for _ in range(n_boot)])
        ci_lo, ci_hi = numpy.percentile(boot_meds, [2.5, 97.5])
        rb = 1 - (2 * stat) / (n * (n + 1) / 2)

        results.append({
            'predictor': pred,
            'pred_label': utils.env_variable_no_unit_label_dict[pred],
            'comparison': pair_name,
            'n_asv': n,
            'n_alt_greater': (diff > 0).sum(),
            'n_dna_greater': (diff < 0).sum(),
            'pct_alt_greater': round(100 * (diff > 0).mean(), 1),
            'median_diff': round(med_diff, 6),
            'ci_lo': round(ci_lo, 6),
            'ci_hi': round(ci_hi, 6),
            'rank_biserial': round(rb, 3),
            'W_stat': stat,
            'p_raw': p,
        })


res = pandas.DataFrame(results)
_, p_adj, _, _ = multipletests(res['p_raw'], method='fdr_bh')
res['p_adj_BH'] = p_adj
res['sig'] = p_adj < alpha
res['direction'] = res.apply(lambda r: ('alt > DNA' if r['median_diff'] > 0 else 'DNA > alt') if r['sig'] else 'n.s.', axis=1)

dna_iqr = (
    df_sens_triplets[df_sens_triplets['dtype'] == 'dna'].groupby('predictor')['mean_abs_deriv'].quantile(0.75)
    - df_sens_triplets[df_sens_triplets['dtype'] == 'dna'].groupby('predictor')['mean_abs_deriv'].quantile(0.25)
).rename('iqr_dna').reset_index()

res = res.merge(dna_iqr, on='predictor', how='left')
res['iqr_dna'] = res['iqr_dna'].replace(0, numpy.nan)
res['scaled_diff'] = res['median_diff'] / res['iqr_dna']
res['scaled_ci_lo'] = res['ci_lo'] / res['iqr_dna']
res['scaled_ci_hi'] = res['ci_hi'] / res['iqr_dna']

sig_preds_rna_dna = (res[(res['comparison'] == pair_rnadna) & res['sig'] & (res['median_diff'] > 0)]['predictor'].tolist())
pandas.DataFrame({'predictor': sig_preds_rna_dna}).to_csv('%sgam_results/sig_preds_rna_dna.csv' % config.data_directory, index=False)

# Rank predictors by rRNA:rDNA scaled_diff
pred_rank = (res[res['comparison'] == pair_rnadna].set_index('predictor')['scaled_diff'].sort_values().index.tolist())
pred_rank = [p for p in pred_rank if p in utils.env_variable_to_plot]
n_pred  = len(pred_rank)

pairs = [pair_rna, pair_rnadna]
offsets = {pair_rna: 0.18, pair_rnadna: -0.18}



ax_r2.axvline(0, color='k', lw=2.5, ls=':', zorder=0)
x_max = numpy.percentile(res['scaled_ci_hi'].dropna(), 90)
x_min = res['scaled_ci_lo'].min()
ax_r2.set_xlim(x_min - 0.05, x_max + 0.1)

for pi, pred in enumerate(pred_rank[::-1]):
        for pair in pairs:
            row = res[(res['predictor'] == pred) & (res['comparison'] == pair)].iloc[0]
            y   = pi + offsets[pair]
            x   = row['scaled_diff']
            lo  = row['scaled_ci_lo']
            hi  = row['scaled_ci_hi']
            col = pairs_colors[pair]
            sig = row['sig']

            ax_r2.plot([lo, hi], [y, y], color=col, lw=1.5, zorder=2, solid_capstyle='round')
            for xv in [lo, hi]:
                ax_r2.plot([xv, xv], [y - 0.07, y + 0.07], color=col, lw=1.2, zorder=2)
            fc = col if sig else 'white'
            ax_r2.scatter(x, y, color=fc, s=55, zorder=4, edgecolors=col, linewidths=1.2)



# y-axis: colored ticks + bracket per predictor
# y-axis: two colored tick marks per predictor (no vertical bar)
# label centred between them, separator lines between predictor groups
ax_r2.set_yticks([])
ax_r2.tick_params(axis='y', length=0)
ax_r2.invert_yaxis()

xlim = ax_r2.get_xlim()
xrange = xlim[1] - xlim[0]
tick_len = 0.018 * xrange
# left end of colored ticks
tick_end = xlim[0] - tick_len
# label x position
label_x  = tick_end - 0.015 * xrange

for pi, pred in enumerate(pred_rank[::-1]):
    y_rna = pi + offsets[pair_rna]
    y_rnadna = pi + offsets[pair_rnadna]
    # exact midpoint between the two ticks
    y_mid = pi

    # Colored tick for rRNA
    ax_r2.plot([xlim[0], tick_end], [y_rna, y_rna], color=pairs_colors[pair_rna], lw=2, clip_on=False, zorder=5, solid_capstyle='butt')

    # Colored tick for rRNA:rDNA
    ax_r2.plot([xlim[0], tick_end], [y_rnadna, y_rnadna], color=pairs_colors[pair_rnadna], lw=2, clip_on=False, zorder=5, solid_capstyle='butt')

    # Label centred between the two ticks.
    # rotation_mode='anchor' rotates around the ha/va anchor point
    # so the vertical centre stays locked to y_mid regardless of rotation.
    ax_r2.text(label_x, y_mid, utils.env_variable_no_unit_label_abbrev_dict[pred], ha='right', va='center', fontsize=8, rotation=45, rotation_mode='anchor', clip_on=False)

# Dotted separator lines between each pair of environmental variables
for pi in range(n_pred - 1):
    ax_r2.axhline(pi + 0.5, color='k', lw=1, ls=':', zorder=1, alpha=1)

ax_r2.set_xlabel('Median scaled difference in sensitivity\nrelative to rDNA', fontsize=11)

legend_elements = [
    mpatches.Patch(color=pairs_colors[pair_rna],    label='rRNA'),
    mpatches.Patch(color=pairs_colors[pair_rnadna], label='rRNA:rDNA'),
    plt.scatter([], [], color='grey',  s=45, label=r'$P < 0.05$ (BH-FDR)'),
    plt.scatter([], [], color='white', s=45, edgecolors='grey', linewidths=1.2, label=r'$P \nless 0.05$')
]

# mpatches.Patch(color='grey', alpha=0.15, label=r'$\bar{\delta}_j(x)$: $\pm$1 SE'),
ax_r2.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=5.5, framealpha=1)
ax_r2.xaxis.set_tick_params(labelsize=6)

fig_name = "%stime_vs_env_and_gam_sensitivity.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



# statistical test for evenness
def per_asv_gradient_shape(df_deriv, pair_name, pred):
    alt_dtype = pair_dtype_map[pair_name]

    sub = df_deriv[df_deriv['predictor'] == pred].copy()
    dna_d = sub[sub['dtype']== 'dna'][['asv_id','x_value','derivative']].rename(columns={'derivative':'deriv_dna'})
    alt_d = sub[sub['dtype'] == alt_dtype][['asv_id','x_value','derivative']].rename(columns={'derivative':'deriv_alt'})
    merged = pandas.merge(dna_d, alt_d, on=['asv_id','x_value'])
    merged['delta'] = merged['deriv_alt'] - merged['deriv_dna']

    records = []
    for asv, grp in merged.groupby('asv_id'):
        rho, p = stats.spearmanr(grp['x_value'], grp['delta'])
        records.append({'asv_id': asv, 'spearman_r': rho, 'p': p})

    return pandas.DataFrame(records).sort_values('spearman_r', ascending=False)


def run_per_asv_gradient_shape():

    df_deriv = pandas.read_csv(deriv_curves_path)
    records = []
    for pair_name, alt_dtype in pair_dtype_map.items():
        for pred in utils.env_variable_to_plot:
            sub = df_deriv[df_deriv['predictor'] == pred].copy()

            dna_d = (sub[sub['dtype'] == 'dna'][['asv_id','x_value','derivative']].rename(columns={'derivative':'deriv_dna'}))
            alt_d = (sub[sub['dtype'] == alt_dtype]
                    [['asv_id','x_value','derivative']]
                    .rename(columns={'derivative':'deriv_alt'}))

            merged = pandas.merge(dna_d, alt_d, on=['asv_id','x_value'])
            merged['delta'] = merged['deriv_alt'] - merged['deriv_dna']

            for asv, grp in merged.groupby('asv_id'):
                rho, p = stats.spearmanr(grp['x_value'], grp['delta'])
                records.append({
                    'asv_id':     asv,
                    'predictor':  pred,
                    'comparison': pair_name,
                    'spearman_r': round(rho, 4),
                    'p_raw':      p,
                })

    df_rho = pandas.DataFrame(records)

    # BH-FDR within each comparison
    for pair_name in [pair_rna, pair_rnadna]:
        mask = df_rho['comparison'] == pair_name
        _, p_adj, _, _ = multipletests(df_rho.loc[mask,'p_raw'], method='fdr_bh')
        df_rho.loc[mask, 'p_adj_BH'] = p_adj

    df_rho['sig_BH']  = df_rho['p_adj_BH'] < 0.05
    df_rho['sig_raw'] = df_rho['p_raw']    < 0.05

    print(df_rho.groupby(['comparison','predictor'])['spearman_r'].describe().round(3))

    df_rho.to_csv('%sgam_results/per_asv_gradient_shape_rho.csv' % config.data_directory, index=False)



def asv_contributions(pred, pair_name):
    sub = df_sens_triplets[df_sens_triplets['predictor'] == pred]
    alt_dt = pair_dtype_map[pair_name]
    dna = sub[sub['dtype'] == 'dna'].set_index('asv_id')['mean_abs_deriv']
    alt = sub[sub['dtype'] == alt_dt].set_index('asv_id')['mean_abs_deriv']
    idx = dna.index.intersection(alt.index)
    diff = (alt[idx] - dna[idx]).sort_values(ascending=False)

    return pandas.DataFrame({
        'asv_id':    diff.index,
        'diff':      diff.values,
        'direction': numpy.where(diff.values > 0, 'alt > rDNA', 'rDNA > alt'),
    })



def rank_contributions(pred, pair_name):
    sub = df_sens_triplets[df_sens_triplets['predictor'] == pred]
    alt_dt = pair_dtype_map[pair_name]
    dna = sub[sub['dtype'] == 'dna'].set_index('asv_id')['mean_abs_deriv']
    alt = sub[sub['dtype'] == alt_dt].set_index('asv_id')['mean_abs_deriv']
    idx = dna.index.intersection(alt.index)
    d = alt[idx] - dna[idx]

    ranks = stats.rankdata(numpy.abs(d.values))
    # signed rank per ASV
    W_contrib = numpy.sign(d.values) * ranks
    W_total = numpy.abs(W_contrib).sum()

    return (pandas.DataFrame({
        'asv_id': idx,
        'diff': d.values.round(5),
        'W_contrib': W_contrib.round(1),
        'pct_W': (numpy.abs(W_contrib) / W_total * 100).round(1),
    }).sort_values('W_contrib', ascending=False).reset_index(drop=True))


def leave_one_out(pred, pair_name):
    
    sub = df_sens_triplets[df_sens_triplets['predictor'] == pred]
    alt_dt = pair_dtype_map[pair_name]
    dna = sub[sub['dtype'] == 'dna'].set_index('asv_id')['mean_abs_deriv']
    alt = sub[sub['dtype'] == alt_dt].set_index('asv_id')['mean_abs_deriv']
    idx = dna.index.intersection(alt.index)
    d = (alt[idx] - dna[idx]).values
    asvs = idx.tolist()

    _, p_full = stats.wilcoxon(d, alternative='two-sided', zero_method='wilcox')
    med_full = numpy.median(d)

    records = []
    for k, asv in enumerate(asvs):
        mask  = numpy.ones(len(d), dtype=bool)
        mask[k] = False
        d_loo = d[mask]
        _, p_loo = stats.wilcoxon(d_loo, alternative='two-sided', zero_method='wilcox')
        records.append({
            'asv_id': asv,
            'diff': round(d[k], 5),
            'med_loo': round(numpy.median(d_loo), 5),
            'delta_med': round(numpy.median(d_loo) - med_full, 5),
            'p_loo': round(p_loo, 4),
            'changes_sig': (p_loo >= 0.05) != (p_full >= 0.05),
        })

    return (pandas.DataFrame(records).sort_values('diff', ascending=False).reset_index(drop=True))




sig_combos = res[res['sig']][['predictor','comparison']].values.tolist()

for pred, pair_name in sig_combos:
    print(f'\n{"="*60}')
    print(f'{pred}  |  {pair_name}')
    print(f'{"="*60}')

    rc = rank_contributions(pred, pair_name)
    print('\nTop contributors (% of W):')
    print(rc.head(7).to_string(index=False))

    loo = leave_one_out(pred, pair_name)
    flipped = loo[loo['changes_sig']]
    if len(flipped):
        print(f'\nASVs that flip significance when removed:')
        print(flipped[['asv_id','diff','p_loo']].to_string(index=False))
    else:
        print('\nAll ASVs: result robust to LOO removal')