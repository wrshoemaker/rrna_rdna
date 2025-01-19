import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.stats.multitest import fdrcorrection

import sine_parameter_utils




env_variable_all_nested = sine_parameter_utils.env_variable_all_nested
gam_coeff_dict = utils.build_gam_coeff_dict()

otu_list = list(gam_coeff_dict.keys())
otu_list.sort()

y_axis_idx = numpy.asarray(range(len(otu_list)))[::-1]



def plot_gam_stat(stat='p_value_fdr'):


    fig = plt.figure(figsize = (9, 9))
    fig.subplots_adjust(bottom= 0.15)

    for nested_i_idx, nested_i in enumerate(env_variable_all_nested):

        for env_variable_j_idx, env_variable_j in enumerate(nested_i):

            ax = plt.subplot2grid((3, 3), (nested_i_idx, env_variable_j_idx), colspan=1)

            for data_type in ['dna', 'rna']:

                otu_stat = numpy.asarray([gam_coeff_dict[k][data_type][env_variable_j][stat] for k in otu_list])

                # FDR correct for p-values
                if stat == 'p_value_fdr':
                    #otu_stat = fdrcorrection(otu_stat, alpha=0.05, method='indep', is_sorted=False)[1]
                    otu_stat = -1*numpy.log10(otu_stat)

                # number significant
                n_significant = sum(otu_stat > -1*numpy.log10(0.05))

                print(env_variable_j, data_type, n_significant)

                ax.scatter(otu_stat, y_axis_idx, alpha=0.7, s=30, color=utils.dna_rna_color_dict[data_type.upper()], label=data_type.upper())


            ax.set_yticks(y_axis_idx)
            ax.set_yticklabels(otu_list, fontsize=5)


            if stat == 'p_value_fdr':
                ax.set_xlabel(r'$- \mathrm{log}_{10}P$' + ', FDR-corrected', fontsize=10)
            
            ax.axvline(x=-1*numpy.log10(0.05), lw=2.5, ls=':', label=r'$P = 0.05$', color='k', zorder=1)
            ax.set_title(utils.env_variable_label_dict[env_variable_j], fontsize=10)
            ax.set_xlim([-0.1,9.1])

            if (env_variable_j_idx == 0) and (nested_i_idx == 0):
                ax.legend(loc='lower right', fontsize=6)


    fig.subplots_adjust(hspace=0.4, wspace=0.30)
    fig_name = "%sgam_%s.png" % (config.analysis_directory, stat)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


plot_gam_stat()
