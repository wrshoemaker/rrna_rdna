
import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
# numdifftools also installed
import pickle

import difference_utils


derivative_order_all = [1,2]
derivative_order_label_all = ['First', 'Second']

minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()

metadata_dict = utils.build_metadata_dict()

s_by_s, otu_labels, samples = utils.load_count_data()
s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

# get days
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

def plot_env_derivative(env_variable):

    env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])

    # remove nan in environmental variable
    to_keep_idx = numpy.isfinite(env_variable_array)
    env_variable_array = env_variable_array[to_keep_idx]
    
    days_env = days[to_keep_idx]
    s_by_s_dna_env = s_by_s_dna[:,to_keep_idx]
    s_by_s_rna_env = s_by_s_rna[:,to_keep_idx]

    # rescale
    rel_s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna_env)
    rel_s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna_env)
    rel_s_by_s_rescaled_ratio = rel_s_by_s_rescaled_rna/rel_s_by_s_rescaled_dna

    rel_s_by_s_rescaled_dna_log = numpy.log10(rel_s_by_s_rescaled_dna)
    rel_s_by_s_rescaled_rna_log = numpy.log10(rel_s_by_s_rescaled_rna)
    rel_s_by_s_rescaled_ratio_log = numpy.log10(rel_s_by_s_rescaled_ratio)

    rel_s_by_s_rescaled_log_dict = {'DNA':rel_s_by_s_rescaled_dna_log, 'RNA':rel_s_by_s_rescaled_rna_log, 'ratio':rel_s_by_s_rescaled_ratio_log}


    fig = plt.figure(figsize = (12, 8))
    fig.subplots_adjust(bottom= 0.15)

    for derivative_order_idx, derivative_order in enumerate(derivative_order_all):

        env_derivative = difference_utils.fd_derivative(env_variable_array, days_env, n=derivative_order, m=3)
        #env_derivative = numpy.diff(env_variable, derivative_order)/numpy.diff(days, derivative_order)

        for data_type_idx, data_type in enumerate(utils.data_type_all):

            ax = plt.subplot2grid((2, 3), (derivative_order_idx, data_type_idx))

            rel_s_by_s_rescaled_log = rel_s_by_s_rescaled_log_dict[data_type]
            for otu_i_idx in range(rel_s_by_s_rescaled_dna.shape[0]):

                otu_derivative = difference_utils.fd_derivative(rel_s_by_s_rescaled_log[otu_i_idx,:], days_env, n=derivative_order, m=3)
                ax.plot(days_env[2:], otu_derivative[2:], lw=1, ls='-', alpha=0.4, c=utils.dna_rna_color_dict[data_type])

            ax.plot(days_env[2:], env_derivative[2:], lw=2, ls='--', c='k', zorder=3, label=utils.env_variable_label_dict[env_variable])
            #ax.plot(days[:-derivative_order], env_derivative, lw=2, ls='--', c='k', zorder=3)

            ax.set_xlabel('Time (days)', fontsize=11)
            ax.set_ylabel('%s-order derivative' % derivative_order_label_all[derivative_order_idx], fontsize=11)
            ax.set_title(utils.sample_label_dict[data_type], fontsize=12)
            
            # ticks
            ax.set_xlim([0, max(days)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)
            ax.yaxis.set_tick_params(labelsize=7)


            if (derivative_order_idx==0) and (data_type_idx==0):
                ax.legend(loc='upper left', fontsize=8)



    fig.subplots_adjust(hspace=0.35,wspace=0.25)
    fig_name = "%senv_derivative/env_derivative_%s.png" % (config.analysis_directory, env_variable)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




for e in utils.env_variables_all:

    plot_env_derivative(e)




