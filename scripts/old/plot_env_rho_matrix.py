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
param_env_dict = sine_parameter_utils.load_param_env_dict()

s_by_s, otu_labels, samples = utils.load_count_data()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


env_variable_all = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph', 'total_nitrogen', 'total_phosphorus', 'doc']
env_variable_dict = {}
to_keep_idx = numpy.full((len(days)), True)

for env_variable in env_variable_all:

    env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])
    # remove nans
    #env_to_keep_idx = (~numpy.isnan(env_variable_array))
    #env_variable_array_clean = env_variable_array[env_to_keep_idx]
    #days_clean = days[env_to_keep_idx]
    # find missing data
    to_keep_idx *= to_keep_idx*(~numpy.isnan(env_variable_array))
    env_variable_dict[env_variable] = env_variable_array


env_variable_array_all = [env_variable_dict[e][to_keep_idx] for e in env_variable_all]
env_variable_matrix = numpy.vstack(env_variable_array_all)
env_variable_rho = numpy.corrcoef(env_variable_matrix)
env_variable_rho[numpy.tril_indices(env_variable_rho.shape[0], 0)] = numpy.nan


fig, ax = plt.subplots(figsize=(5.5,5))
dummy_range = range(len(env_variable_all)+1)
pcm = ax.pcolor(dummy_range, dummy_range, env_variable_rho, cmap='coolwarm', norm=colors.Normalize(vmin=-1, vmax=1))
pcm.cmap.set_under('black')            

#ax.set_xlabel("# secreted resources, BHI", fontsize=7)
#ax.set_ylabel("# secreted resources, M9", fontsize=7)
#ax.set_title("Fract. C secreted = %.2f\nfract. BHI secreted = %.2f" % (l_c, l_bhi), fontsize=7)
# color range, 0 - 0.1

tick_idx = numpy.arange(len(env_variable_all))+0.5

env_variable_name = [utils.env_variable_no_unit_label_split_dict[e] for e in env_variable_all]

ax.set_xticks(tick_idx)
ax.set_xticklabels(env_variable_name, fontsize=5)

ax.set_yticks(tick_idx)
ax.set_yticklabels(env_variable_name, fontsize=5, rotation=90, rotation_mode='anchor', ha="center")
ax.tick_params(axis='y', which='major', pad=15)



#if (l_c_idx == 0) and (l_bhi_idx == 0):
clb_slope = plt.colorbar(pcm, ax=ax)
clb_slope.set_label(label='Correlation' , fontsize=7)
clb_slope.ax.tick_params(labelsize=7)




fig.subplots_adjust(hspace=0.15, wspace=0.15)
fig_name = "%senv_corr_heatmap.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



