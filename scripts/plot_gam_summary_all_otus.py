import config
import sys
import random
import argparse
import copy
import numpy
import utils
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy import stats, signal
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from statsmodels.stats.multitest import fdrcorrection

import sine_parameter_utils

# numdifftools also installed
import pickle
from scipy.stats import gamma, loggamma, nbinom, norm
import plot_predict_change_dna

stat = 'p_value_fdr'


gam_dict = utils.build_gam_coeff_dict()
env_variable_all = list(gam_dict[(list(gam_dict.keys())[0])]['dna'].keys())
env_variable_all.sort()


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
taxonomy_dict = utils.build_taxonomy_dict()
param_dict =  pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, 'rb'))
otu_labels_param = param_dict['otu_labels']


env_variable_label_all = numpy.asarray([utils.env_variable_no_unit_label_dict[e] for e in env_variable_all])
#sort_idx = numpy.argsort(numpy.asarray([gam_dict[e]['dna']['coeff_scaled'] for e in env_variable_all]))[::-1]
#env_variable_label_all = env_variable_label_all[sort_idx]



def make_plot(stat):

    if stat == 'p_value_fdr':
        x_line = -1*numpy.log10(0.05)
    else:
        x_line = 0


    idx_all = list(range(len(otu_labels_param)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    asv_count = 0
    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):
            
            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            otu_label_c = otu_labels_param[asv_count]

            for data_type in ['dna', 'rna', 'rna_dna']:
                stat_all = numpy.asarray([gam_dict[otu_label_c][data_type][e][stat] for e in env_variable_all])

                print(stat_all)
                if stat == 'p_value_fdr':
                    stat_all = -1*numpy.log10(stat_all)
                    ax.set_xlabel(r'$- \mathrm{log}_{10}\,P$' + ', FDR-corrected', fontsize=10)
                    #ax.set_xlim([-0.2, 3])

                else:
                    ax.set_xlabel('GAM coefficient * std. deviation', fontsize=10)
                    #ax.set_xlim([-0.2, 3.5])

                ax.scatter(stat_all, range(len(stat_all)), zorder=2, alpha=0.7, s=30, color=utils.dna_rna_color_dict[data_type.upper()], label=utils.sample_label_dict[data_type.upper()])
                ax.axvline(x=x_line, lw=2.5, ls=':', color='k', zorder=1)

                ax.set_yticks(range(len(env_variable_label_all)))
                ax.set_yticklabels(env_variable_label_all, fontsize=6, rotation=45)

                ax.set_title('ASV %d (%s)' % (asv_count+1, taxonomy_dict[otu_label_c]['family']), fontsize=11)

                if asv_count == 0:
                    ax.legend(loc='lower right', fontsize=6)


            asv_count += 1



    fig.subplots_adjust(hspace=0.4, wspace=0.35)
    fig_name = "%sgamma_summary_all_otus_%s.png" % (config.analysis_directory, stat)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



make_plot('coeff')
make_plot('p_value_fdr')