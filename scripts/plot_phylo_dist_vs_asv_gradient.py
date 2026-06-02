import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from scipy import stats, cluster, spatial
from statsmodels.stats.multitest import multipletests
import pickle
from itertools import combinations  

import config
import utils
import sine_parameter_utils


#data_type = 'rna'
#data_type = 'rna'

fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(bottom= 0.15)

phototroph_all = ['ASV_001', 'ASV_005']

#'TACGGGGGATGCAAGCGTTATCCGGAATGATTGGGCGTAAAGAGTCCGTAGGTAGTCATCCAAGTCTGCTGTTAAAGAGCGAGGCTTAACCTCGTAAAGGCAGTGGAAACTGGAAGACTAGAGTGTAGTAGGGGCAGAGGGAATTCCTGGTGTAGCGGTGAAATGCGTAGAGATCAGGAAGAACACCGGTGGCGAAGGCGCTCTGCTGGGCTATAACTGACACTGAGGGACGAAAGCTAGGGGAGCGAATGGG'
#param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
#print(param_dict['otu_labels'].index('TACGGGGGATGCAAGCGTTATCCGGAATGATTGGGCGTAAAGAGTCCGTAGGTAGTCATCCAAGTCTGCTGTTAAAGAGCGAGGCTTAACCTCGTAAAGGCAGTGGAAACTGGAAGACTAGAGTGTAGTAGGGGCAGAGGGAATTCCTGGTGTAGCGGTGAAATGCGTAGAGATCAGGAAGAACACCGGTGGCGAAGGCGCTCTGCTGGGCTATAACTGACACTGAGGGACGAAAGCTAGGGGAGCGAATGGG'))

PRED_ORDER = utils.env_variable_to_plot
PRED_LABELS = utils.env_variable_no_unit_label_dict
deriv_curves_path = '%sgam_results/08_derivative_curves.csv' % config.data_directory
df_deriv = pd.read_csv(deriv_curves_path)

df_asv = pd.read_csv("%sgam_results/asv_id_lookup.csv" % config.data_directory)
df_asv["asv_seq"] = df_asv["base_name"].str.split("_").str[1]

dist_dict = pickle.load(open(config.data_directory + 'otu_dist_dict.pickle', "rb"))

def build_similarity_matrix_single_dtype(df_deriv, dtype):

    # Build pairwise Pearson similarity matrix for a single data type
    # based on its raw pointwise derivative curves.

    # For each predictor j and ASV i:
    #   feature = d f_{j,g,i}/dx(x)  (signed derivative, standardised)

    sub_dtype = df_deriv[df_deriv['dtype'] == dtype].copy()
    asv_list  = sorted(sub_dtype['asv_id'].unique())

    feature_rows = []

    for pred in PRED_ORDER:
        sub = (sub_dtype[sub_dtype['predictor'] == pred][['asv_id', 'x_value', 'derivative']].copy())

        # Standardise within this predictor so all predictors
        # contribute equally regardless of scale
        mu  = sub['derivative'].mean()
        std = sub['derivative'].std(ddof=1)
        sub['deriv_std'] = (sub['derivative'] - mu) / std if std > 0 else 0.0

        # Pivot: rows = ASV, cols = x_value
        piv = (sub.pivot(index='asv_id', columns='x_value', values='deriv_std').reindex(index=asv_list).sort_index(axis=1))
        feature_rows.append(piv)

    # Concatenate across predictors: shape (n_asv, n_pred × n_x)
    feat_df = pd.concat(feature_rows, axis=1)
    feat_df.columns = range(feat_df.shape[1])

    sim_matrix = np.corrcoef(feat_df.values)

    return sim_matrix, asv_list, feat_df


def cluster_order(sim_matrix, asv_list):
    dissim = 1.0 - sim_matrix
    np.fill_diagonal(dissim, 0)

    # Enforce exact symmetry
    dissim = (dissim + dissim.T) / 2
    dissim = np.clip(dissim, 0, None)

    condensed = spatial.distance.squareform(dissim)
    linkage   = cluster.hierarchy.ward(condensed)
    dendro    = cluster.hierarchy.dendrogram(linkage, no_plot=True)
    order     = dendro['leaves']
    return [asv_list[i] for i in order]




for data_type_idx, data_type in enumerate(['dna', 'rna']):

    #df_matrix = pd.read_csv("%sgam_results/asv_similarity_%s.csv" % (config.data_directory, data_type), index_col=0)
    sim, asv_list, feat = build_similarity_matrix_single_dtype(df_deriv, data_type)
    asv_ordered = cluster_order(sim, asv_list)
    #results[dtype] = {'sim': sim, 'asv_list':  asv_list, 'asv_ordered': asv_ordered}
    upper = sim[np.triu_indices(len(asv_list), k=1)]

    sim_df = pd.DataFrame(sim, index=asv_list, columns=asv_list)
    asv_all = sim_df.index.tolist()
    asv_pairs_all = list(combinations(asv_all, 2))

    # dist for pairs
    pairs = []
    dist_all = []
    for i, r in enumerate(sim_df.index):
        for j, c in enumerate(sim_df.columns):
            
            if j <= i:
                continue

            val = sim_df.loc[r, c]
            n_selected = (r in phototroph_all) + (c in phototroph_all)
            pairs.append({"row": r, "col": c, "value": val, "n_selected": n_selected})

            seq_1 = df_asv.loc[df_asv["asv_id"] == r, "asv_seq"].iloc[0]
            seq_2 = df_asv.loc[df_asv["asv_id"] == c, "asv_seq"].iloc[0]

            if (seq_1, seq_2) in dist_dict:
                dist_12 = dist_dict[(seq_1, seq_2)]
            else:
                dist_12 = dist_dict[(seq_2, seq_1)]

            dist_all.append(dist_12)

    pairs = pd.DataFrame(pairs)
    pairs_all = pairs['value'].values
    dist_all = np.asarray(dist_all)

    idx_photo_photo = pairs.n_selected == 2
    idx_photo_hetero = pairs.n_selected == 1
    idx_hetero_hetero = pairs.n_selected == 0

    photo_photo_rho = pairs[idx_photo_photo]['value'].values
    photo_hetero_rho = pairs[idx_photo_hetero]['value'].values
    hetero_hetero_rho = pairs[idx_hetero_hetero]['value'].values


    color = utils.dna_rna_color_dict[data_type.upper()]

    ax_dist = plt.subplot2grid((2, 2), (0, data_type_idx))
    ax_decay = plt.subplot2grid((2, 2), (1, data_type_idx))

    # plot distributions
    #ax_dist.hist(pairs_all[idx_hetero_hetero], bins=12, density=True, histtype='step', alpha=1, lw=4, ls=':', color=color, zorder=1, label='hetero x hetero')
    #ax_dist.hist(pairs_all[idx_photo_hetero], bins=12, density=True, histtype='step', alpha=1, lw=4, ls='--', color=color, zorder=1, label='photo x hetero')

    counts_hh, edges_hh = np.histogram(pairs_all[idx_hetero_hetero], bins=9, density=True)
    ax_dist.stairs(counts_hh, edges_hh, color=color, linewidth=2.5, linestyle='-', label='hetero x hetero')

    counts_ph, edges_ph = np.histogram(pairs_all[idx_photo_hetero], bins=9, density=True)
    ax_dist.stairs(counts_ph, edges_ph, color=color, linewidth=2.5, linestyle='--', label='photo x hetero')

    ax_dist.axvline(x=pairs_all[idx_photo_photo][0], lw=4, ls=':', color=color, label='photo x photo')

    #counts_hh, edges_hh = np.histogram(pairs_all[idx_hetero_hetero], bins=12, density=True)
    #ax_dist.stairs(counts_hh, edges_hh, color=color, linewidth=2, linestyle='-')
    ax_dist.set_xlabel('Corr. in sensitivity to env. variables', fontsize=12)
    ax_dist.set_ylabel('Probability density', fontsize=12)

    legend_elements_dist = [
    Line2D([0], [0], color=color, lw=2.5, ls='-',  label='hetero \u00d7 hetero'),
    Line2D([0], [0], color=color, lw=2.5, ls='--', label='photo \u00d7 hetero'),
    Line2D([0], [0], color=color, lw=4,   ls=':',  label='photo \u00d7 photo')]

    #if data_type_idx == 0:
    ax_dist.legend(handles=legend_elements_dist, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, columnspacing=0.7, handletextpad=0.6)


    marker_style_photo_hetero = dict(color='k', marker='o', markerfacecoloralt=color, fillstyle='left', markerfacecolor='white', mew=1)
    #marker_style_hetero_hetero = dict(color='k', marker='o', markerfacecoloralt='white', markerfacecolor='white', mew=1)
    #marker_style_photo_photo = dict(color='k', marker='o', markerfacecoloralt=color, markerfacecolor=color, mew=1)
    marker_style_hetero_hetero = dict(color='k', marker='o', markerfacecolor='white', fillstyle='full', mew=1)
    marker_style_photo_photo = dict(color='k', marker='o', markerfacecolor=color, fillstyle='full', mew=1)
    #ax_decay.scatter(dist_all, pairs, s=200, facecolor='white', edgecolor=color, linewidth=1)

    for i in range(sum(idx_hetero_hetero)):
        ax_decay.plot(dist_all[idx_hetero_hetero][i], pairs_all[idx_hetero_hetero][i], markersize=7, linewidth=1, alpha=1, zorder=3, **marker_style_hetero_hetero)

    for i in range(sum(idx_photo_hetero)):
        ax_decay.plot(dist_all[idx_photo_hetero][i], pairs_all[idx_photo_hetero][i], markersize=7, linewidth=1, alpha=1, zorder=3, **marker_style_photo_hetero)

    for i in range(sum(idx_photo_photo)):
        ax_decay.plot(dist_all[idx_photo_photo][i], pairs_all[idx_photo_photo][i], markersize=7, linewidth=1, alpha=1, zorder=3, **marker_style_photo_photo)

    
    #slope, intercept, r_value, p_value, std_err = stats.linregress(dist_all, pairs_all)
    slope, intercept, r_value, p_value_perm, std_err = utils.perm_slope(dist_all, pairs_all, n_perm=10)
    # permutation test
    x_range = np.linspace(min(dist_all), max(dist_all), num=1000)
    y_pred = intercept + slope*x_range

    if data_type == 'rna':
        x_text_ = 0.73
    else:
        x_text_ = 0.73

    ax_decay.plot(x_range, y_pred, lw=2, ls='--', c='k', zorder=3)
    ax_decay.text(x_text_, 0.92, r'$\rho^{2} = $' + str(round(r_value**2, 3)), fontsize=9, transform=ax_decay.transAxes)
    ax_decay.text(x_text_, 0.84, r'$P < 0.05 $', fontsize=9, transform=ax_decay.transAxes)
    ax_decay.set_ylim([-0.84, 0.82])

    legend_handles = [
    Line2D([0],[0], marker='o', ls='None', mfc=color, color='black', label='photo \u00d7 photo'),
    Line2D([0],[0], marker='o', ls='None', mfc='white', mfcalt=color, fillstyle='left', color='black', label='photo \u00d7 hetero'),
    Line2D([0],[0], marker='o', ls='None', mfc='white', color='black', label='hetero \u00d7 hetero')]

    #if data_type_idx == 0:
    #    #ax_decay.legend(handles=legend_handles, frameon=True, loc='upper right')
    ax_decay.legend(handles=legend_handles, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, columnspacing=0.2, handletextpad=0.1)
    ax_decay.set_xlabel('Pairwise phylogenetic distance', fontsize=12)
    ax_decay.set_ylabel('Corr. in sensitivity to env. variables', fontsize=12)

  

#ax.scatter(dist_all, rho_all)



fig.subplots_adjust(hspace=0.3, wspace=0.35)
fig_name = "%sphylo_dist_vs_asv_gradient_%s.png" % (config.analysis_directory, data_type)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


# df["new_col"] = df["base_name"].str.split("_", n=2).str[1]


#print(df_asv.query("asv_id == %s" % 'ASV_006')["asv_seq"])


#print(dist_dict)

#value = df_matrix.loc[row_label, column_label]