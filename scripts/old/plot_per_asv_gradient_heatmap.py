import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats, cluster, spatial
from statsmodels.stats.multitest import multipletests

import config
import utils

# =============================================================================
# ASV pairwise Pearson similarity heatmap
#
# For each comparison (rRNA vs rDNA, rRNA:rDNA vs rDNA):
#
#   1. Compute delta_{j,i}(x) = deriv_alt(x) - deriv_rDNA(x)
#      for every ASV i, predictor j, evaluation point x
#
#   2. Standardise delta within each predictor (zero mean, unit SD across
#      all ASVs × evaluation points) so all predictors contribute equally
#      regardless of scale
#
#   3. Build feature matrix: ASV × (predictor × x_value), shape (21, ~900)
#
#   4. Compute pairwise Pearson r between rows → 21×21 similarity matrix
#
#   5. Order ASVs by hierarchical clustering (Ward linkage) on 1 - r
#
# Two panels side by side, shared colour scale.
# =============================================================================

deriv_curves_path = '%sgam_results/08_derivative_curves.csv' % config.data_directory

PRED_ORDER  = utils.env_variable_to_plot
PRED_LABELS = utils.env_variable_no_unit_label_dict

pair_rna    = 'rRNA vs rDNA'
pair_rnadna = 'rRNA:rDNA vs rDNA'
pair_dtype_map = {
    pair_rnadna: 'rna_dna',
    pair_rna:    'rna',
}

mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   False,
    'axes.spines.bottom': False,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})


# =============================================================================
# 1. BUILD FEATURE MATRIX AND COMPUTE PEARSON SIMILARITY
# =============================================================================

def build_similarity_matrix(df_deriv, pair_name):
    """
    Returns:
      sim_matrix : (n_asv × n_asv) Pearson r similarity matrix
      asv_list   : ordered list of ASV IDs
      feat_df    : feature matrix (ASV × predictor×x_value) before correlation
    """
    alt_dtype = pair_dtype_map[pair_name]

    feature_rows = []
    asv_list     = sorted(df_deriv['asv_id'].unique())

    for pred in PRED_ORDER:
        sub = df_deriv[df_deriv['predictor'] == pred].copy()

        dna_d = (sub[sub['dtype'] == 'dna']
                 [['asv_id', 'x_value', 'derivative']]
                 .rename(columns={'derivative': 'deriv_dna'}))
        alt_d = (sub[sub['dtype'] == alt_dtype]
                 [['asv_id', 'x_value', 'derivative']]
                 .rename(columns={'derivative': 'deriv_alt'}))

        merged = pandas.merge(dna_d, alt_d, on=['asv_id', 'x_value'])
        merged['delta'] = merged['deriv_alt'] - merged['deriv_dna']

        # Standardise within this predictor so all predictors contribute equally
        mu  = merged['delta'].mean()
        std = merged['delta'].std(ddof=1)
        merged['delta_std'] = (merged['delta'] - mu) / std if std > 0 else 0.0

        # Pivot: rows = ASV, cols = x_value
        piv = (merged.pivot(index='asv_id', columns='x_value', values='delta_std')
               .reindex(index=asv_list)
               .sort_index(axis=1))

        feature_rows.append(piv)

    # Concatenate across predictors: shape (n_asv, n_pred × n_x)
    feat_df = pandas.concat(feature_rows, axis=1)
    feat_df.columns = range(feat_df.shape[1])   # flatten column index

    # Pairwise Pearson r: numpy.corrcoef operates on rows
    sim_matrix = numpy.corrcoef(feat_df.values)

    return sim_matrix, asv_list, feat_df


def cluster_order(sim_matrix, asv_list):
    dissim = 1.0 - sim_matrix
    numpy.fill_diagonal(dissim, 0)

    # Enforce exact symmetry — corrcoef can produce tiny asymmetries
    dissim = (dissim + dissim.T) / 2
    dissim = numpy.clip(dissim, 0, None)   # numerical safety

    condensed = spatial.distance.squareform(dissim)
    linkage   = cluster.hierarchy.ward(condensed)
    dendro    = cluster.hierarchy.dendrogram(linkage, no_plot=True)
    order     = dendro['leaves']
    return [asv_list[i] for i in order]


# =============================================================================
# 2. DRAW ONE SIMILARITY PANEL
# =============================================================================

def draw_sim_panel(ax, sim_matrix, asv_order, all_asvs, cmap, norm,
                   show_xlabels=True, show_ylabels=True, title=''):

    n = len(asv_order)
    idx = [all_asvs.index(a) for a in asv_order]
    mat = sim_matrix[numpy.ix_(idx, idx)]

    for ri in range(n):
        for ci in range(n):
            val   = mat[ri, ci]
            color = cmap(norm(val))
            rect  = plt.Rectangle(
                [ci - 0.5, ri - 0.5], 1, 1,
                facecolor=color, edgecolor='white', lw=0.3
            )
            ax.add_patch(rect)

            # Only annotate off-diagonal cells where space allows
            if ri != ci:
                r_, g_, b_, _ = color
                lum = 0.299*r_ + 0.587*g_ + 0.114*b_
                tc  = 'white' if lum < 0.55 else '#333'
                ax.text(ci, ri, f'{val:.2f}',
                        ha='center', va='center',
                        fontsize=5.5, color=tc)
            else:
                # Diagonal: ASV name
                r_, g_, b_, _ = color
                lum = 0.299*r_ + 0.587*g_ + 0.114*b_
                tc  = 'white' if lum < 0.55 else '#333'
                ax.text(ci, ri, asv_order[ri],
                        ha='center', va='center',
                        fontsize=5, color=tc, fontweight='bold')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if show_xlabels:
        ax.set_xticklabels(asv_order, rotation=90, fontsize=7)
        ax.tick_params(axis='x', length=0, pad=2)
    else:
        ax.set_xticklabels([])

    if show_ylabels:
        ax.set_yticklabels(asv_order, fontsize=7)
        ax.tick_params(axis='y', length=0, pad=2)
    else:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)


# =============================================================================
# 3. MAIN FIGURE
# =============================================================================

def make_asv_similarity_heatmap():

    print('Loading derivative curves...')
    df_deriv = pandas.read_csv(deriv_curves_path)

    results = {}
    for pair_name in [pair_rnadna, pair_rna]:
        print(f'Building similarity matrix: {pair_name}...')
        sim, asv_list, feat = build_similarity_matrix(df_deriv, pair_name)
        asv_ordered          = cluster_order(sim, asv_list)
        results[pair_name]   = {
            'sim':         sim,
            'asv_list':    asv_list,
            'asv_ordered': asv_ordered,
        }

        # Summary
        upper = sim[numpy.triu_indices(len(asv_list), k=1)]
        print(f'  Pearson r — mean: {upper.mean():.3f}  '
              f'min: {upper.min():.3f}  max: {upper.max():.3f}')

    # Use same ASV order (from rRNA:rDNA clustering) for both panels
    common_order = results[pair_rnadna]['asv_ordered']

    # Shared colour scale — symmetric around 0
    all_vals = numpy.concatenate([
        results[p]['sim'].flatten() for p in [pair_rnadna, pair_rna]
    ])
    r_max = max(numpy.abs(numpy.percentile(all_vals, [2, 98])).max(), 0.3)
    r_max = min(r_max, 1.0)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'bwr_sim', ['#185FA5', '#f5f4f0', '#A32D2D']
    )
    norm = mcolors.Normalize(vmin=-r_max, vmax=r_max)

    n    = len(common_order)
    cell = 0.45   # cell size in inches

    fig, axes = plt.subplots(
        1, 2,
        figsize=(2 * (n * cell + 0.4), n * cell + 1.2),
        gridspec_kw={'wspace': 0.12}
    )

    draw_sim_panel(
        axes[0],
        results[pair_rnadna]['sim'],
        common_order,
        results[pair_rnadna]['asv_list'],
        cmap, norm,
        show_xlabels=True, show_ylabels=True,
        title='rRNA:rDNA vs rDNA'
    )

    draw_sim_panel(
        axes[1],
        results[pair_rna]['sim'],
        common_order,
        results[pair_rna]['asv_list'],
        cmap, norm,
        show_xlabels=True, show_ylabels=False,
        title='rRNA vs rDNA'
    )

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, shrink=0.8)
    cb.set_label('Pearson r\n(similarity in gradient\nsensitivity profile)',
                 fontsize=8, labelpad=8)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        'ASV pairwise similarity in gradient sensitivity profiles\n'
        r'Pearson $r$ on standardised $\delta_{j,i}(x)$ '
        r'concatenated across all predictors · '
        'ASVs ordered by Ward clustering on $1 - r$',
        fontsize=8.5, y=1.02
    )

    fig_name = '%sfig_asv_similarity_heatmap.png' % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches='tight',
                pad_inches=0.3, dpi=300)
    plt.close()
    print(f'\nSaved: {fig_name}')

    # Save similarity matrices
    for pair_name, key in [(pair_rnadna, 'rnadna'), (pair_rna, 'rna')]:
        r   = results[pair_name]
        out = pandas.DataFrame(
            r['sim'],
            index=r['asv_list'],
            columns=r['asv_list']
        )
        path = '%sgam_results/asv_similarity_%s.csv' % (config.data_directory, key)
        out.to_csv(path)
        print(f'Saved: {path}')


if __name__ == '__main__':
    make_asv_similarity_heatmap()
