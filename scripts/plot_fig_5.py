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
from scipy.stats import gamma, loggamma



metadata_dict = utils.build_metadata_dict()
s_by_s, otu_labels, samples = utils.load_count_data()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

metadata_dict = utils.build_metadata_dict()
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()




fig = plt.figure(figsize = (4, 4)) #
fig.subplots_adjust(bottom= 0.15)

gs = gridspec.GridSpec(nrows=1, ncols=1)

ax_env = fig.add_subplot(gs[0, 0])
ax_afd = ax_env.twinx()


env_variable_array = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples[(sample_type=='RNA')]])
# remove nans
env_to_keep_idx = (~numpy.isnan(env_variable_array))
env_variable_array_clean = env_variable_array[env_to_keep_idx]
env_variable_days_clean = days[env_to_keep_idx]



param_dict =  pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, 'rb'))

days_dna = numpy.asarray(param_dict['data']['days']['DNA'][0])
afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][0])

days_rna = numpy.asarray(param_dict['data']['days']['RNA'][0])
afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][0])


days_inter = numpy.intersect1d(days_dna, env_variable_days_clean)


to_keep_afd_idx = numpy.asarray([numpy.where(days_dna==d)[0][0] for d in days_inter])
to_keep_env_idx = numpy.asarray([numpy.where(env_variable_days_clean==d)[0][0] for d in days_inter])


#env_variable_days_clean = env_variable_days_clean[to_keep_env_idx]
#afd_dna = afd_dna[to_keep_afd_idx]

ax_afd.scatter(days_dna, afd_dna, c=utils.dna_rna_color_dict['DNA'], s=6)
#ax_afd.scatter(days_rna, afd_rna, c=utils.dna_rna_color_dict['RNA'], s=4)
ax_env.scatter(env_variable_days_clean, env_variable_array_clean, c='k', s=6)



ax_afd.set_ylabel("CLR-transformed abundance, DNA", fontsize=11, color=utils.dna_rna_color_dict['DNA'], fontweight='bold')
ax_env.set_ylabel(utils.env_variable_label_dict['water_temp'], fontsize=11, color='k')


ax_env.set_xlim([0, max(days)])
ax_env.set_xticks(minor_days, minor=True)
ax_env.set_xticks(major_days, minor=False)
ax_env.set_xticklabels(major_labels, minor=False, fontsize=7)
ax_env.set_xlabel("Time (days)", fontsize=12)
ax_env.yaxis.set_tick_params(labelsize=7)
ax_afd.yaxis.set_tick_params(labelsize=7)



fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig_name = "%sfig5.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

