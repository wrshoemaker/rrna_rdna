import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from statsmodels.stats.multitest import fdrcorrection

from scipy import stats, signal
# numdifftools also installed
import pickle

import sine_parameter_utils


param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
taxonomy_dict = utils.build_taxonomy_dict()


otu_labels = param_dict['otu_labels']


fig = plt.figure(figsize = (20, 20))
fig.subplots_adjust(bottom= 0.15)

#s_by_s, otu_labels, samples = utils.load_count_data()

fig.suptitle("Gamma: time-varying mean vs. constant mean", fontsize=28, y=0.93)

idx_all = list(range(len(param_dict['data']['clr_afd']['RNA'])))
chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]



asv_count = 0
for chunk_idx, chunk in enumerate(chunk_all):

    for c_idx, c in enumerate(chunk):

        otu_label_i = otu_labels[asv_count]
        ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

        for data_type in ['DNA', 'RNA']:

            lrt_lambda = param_dict['lrt_lambda'][data_type][asv_count]
            lrt_boot_dist = param_dict['lrt_boot_dist'][data_type][asv_count]

            ax.hist(lrt_boot_dist, bins=12, density=True, histtype='step', alpha=1, lw=3, color=utils.dna_rna_color_dict[data_type], zorder=1, label= 'Null, %s' % data_type)
            ax.axvline(x=lrt_lambda, ls=':', c=utils.dna_rna_color_dict[data_type], lw=4, zorder=2, label='Obs., %s' % data_type)


        ax.set_title('ASV %d (%s)' % (asv_count+1, taxonomy_dict[otu_label_i]['family']), fontsize=11)
        ax.xaxis.set_tick_params(labelsize=7)
        ax.yaxis.set_tick_params(labelsize=7)

        ax.set_xlabel('Log-likelihood', fontsize=10)
        ax.set_ylabel('Probability density', fontsize=10)
        

        if asv_count == 0:
            ax.legend(loc='upper left', fontsize=10)


        asv_count += 1


fig.subplots_adjust(hspace=0.4, wspace=0.40)
fig_name = "%slrt_all_otus.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

