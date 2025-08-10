import config
import sys
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

import sine_parameter_utils

# numdifftools also installed
import pickle

import simulation_utils



metadata_dict = utils.build_metadata_dict()
s_by_s, otu_labels, samples = utils.load_count_data()
param_dict =  pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, 'rb'))

otu = 'Otu000001'
otu_idx = param_dict['otu_labels'].index(otu)

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
env_variable_array = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples[(sample_type=='RNA')]])
days_env = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


days_dna = param_dict['data']['days']['DNA'][otu_idx]
afd_dna = param_dict['data']['clr_afd']['DNA'][otu_idx]

days_rna = param_dict['data']['days']['RNA'][otu_idx]
afd_rna = param_dict['data']['clr_afd']['RNA'][otu_idx]


fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)
ax = plt.subplot2grid((1, 1), (0, 0))


ax_temp = ax.twinx()


ax.plot(days_dna, afd_dna, label='DNA', c=utils.dna_rna_color_dict['DNA'], lw=2)
ax.plot(days_rna, afd_rna, label='RNA', c=utils.dna_rna_color_dict['RNA'], lw=2)


ax_temp.plot(days_env[~numpy.isnan(env_variable_array)], env_variable_array[~numpy.isnan(env_variable_array)], c='k', lw=2)



ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel("CLR-transformed abundance\n" + 'OTU 1 ('+ r'$\mathit{Anabaena}$' + ' sp.)', fontsize=12)
ax_temp.set_ylabel(utils.env_variable_label_dict['water_temp'], fontsize=12)


minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()

ax.set_xlim([0, max(days_dna)])
ax.set_xticks(minor_days, minor=True)
ax.set_xticks(major_days, minor=False)
ax.set_xticklabels(major_labels, minor=False, fontsize=7)
ax.legend(loc='lower left')
ax.tick_params(axis='y', labelsize=7)
ax_temp.tick_params(axis='y', labelsize=7)



fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%stime_vs_clr_and_temp.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



