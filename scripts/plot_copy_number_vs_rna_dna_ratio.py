import config
import numpy
import pickle
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.gridspec as gridspec

import sine_parameter_utils

from scipy import stats

taxonomic_level = 'genus'
taxonomy_dict = utils.build_taxonomy_dict()
rrna_copy_dict = utils.make_rrna_copy_dict()
rrna_copy_taxa = numpy.asarray(list(rrna_copy_dict.keys()))

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))

#otu_labels = param_dict[otu_labels'']

use_carrying_capacity_fig_name_dict = {True:'_k', False:''}
use_carrying_capacity_y_label_dict = {True:"Difference in RNA and DNA\nconstants, " + r'$\mathrm{ln} \, K_{\mathrm{RNA}}^{(0)} - \mathrm{ln} \, K_{\mathrm{DNA}}^{(0)}$', False:"Time-averaged RNA:DNA, " + r'$\bar{\phi}_{i}$'}
genus_param = [taxonomy_dict[k][taxonomic_level] for k in param_dict['otu_labels']]

#print(param_dict['otu_labels'])
#print([taxonomy_dict[k]['class'] for k in param_dict['otu_labels']])


def make_plot(use_carrying_capacity):


    to_keep_idx = []
    rrna_copy_number = []
    for g_idx, g in enumerate(genus_param):

        if g in rrna_copy_dict:
            to_keep_idx.append(g_idx)
            rrna_copy_number.append(rrna_copy_dict[g])


    to_keep_idx = numpy.asarray(to_keep_idx)
    mean_ratio_all = []
    for i_idx in to_keep_idx:
        
        param_mean_mle_rna_log = numpy.log(param_dict['param_mean_mle']['RNA'][i_idx])
        param_mean_mle_dna_log = numpy.log(param_dict['param_mean_mle']['DNA'][i_idx])

        clr_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][i_idx])
        clr_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][i_idx])

        #print(param_mean_mle_rna, param_mean_mle_dna)
        #print(numpy.mean(clr_rna))

        if use_carrying_capacity == True:
            diff = param_mean_mle_rna_log - param_mean_mle_dna_log
        else:
            diff = numpy.mean(clr_rna - clr_dna)

        mean_ratio_all.append(diff)


    fig = plt.figure(figsize = (4.5, 4)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=1, ncols=1)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(rrna_copy_number, mean_ratio_all, s=25, alpha=1, color='k', zorder=2, label='One ASV')

    slope, intercept, r_value, p_value, std_err = stats.linregress(rrna_copy_number, mean_ratio_all)


    x_range_ =  numpy.linspace(min(rrna_copy_number), max(rrna_copy_number), 10000)
    y_fit_range = slope*x_range_ + intercept
    ax.plot(x_range_, y_fit_range, ls='--', lw=2.5, c='k')

    x_range_ci, y_range_pred, lcb, ucb = utils.get_confidence_hull(rrna_copy_number, mean_ratio_all)
    idx_to_plot = (x_range_ci >= min(x_range_)) & (x_range_ci <= max(x_range_))
    ax.plot(x_range_ci[idx_to_plot], lcb[idx_to_plot], color='k', linestyle=':', linewidth=2, zorder=3, label=r'$95\%$' + ' confidence hull')
    ax.plot(x_range_ci[idx_to_plot], ucb[idx_to_plot], color='k', linestyle=':', linewidth=2, zorder=3)



    ax.set_xlabel("Mean genus-level rRNA operon copy number", fontsize=12)
    #ax.set_ylabel("Time-averaged RNA:DNA, " + r'$\bar{\phi}_{i}$', fontsize=12)
    ax.set_ylabel(use_carrying_capacity_y_label_dict[use_carrying_capacity], fontsize=12)


    ax.text(0.26, 0.78, r'$\rho^{2} = $' + str(round(r_value**2, 3)), fontsize=11, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.26, 0.69, r'$P = $' + str(round(p_value, 3)), fontsize=11, ha='center', va='center', transform=ax.transAxes)
    

    ax.set_xlim([numpy.min(rrna_copy_number)/1.1, 1.05*numpy.max(rrna_copy_number)])
    #ax.set_ylim([-1.5, 3.5])


    ax.legend(loc='upper left', fontsize=10)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    fig_name = "%scopy_number_vs_rna_dna_ratio%s.png" % (config.analysis_directory, use_carrying_capacity_fig_name_dict[use_carrying_capacity])
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


#print(param_dict['amp_mle'].keys())



if __name__ == "__main__":

    print("Running...")

    make_plot(use_carrying_capacity=True)
    make_plot(use_carrying_capacity=False)