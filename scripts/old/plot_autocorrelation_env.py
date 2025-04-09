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

import sine_parameter_utils


numpy.seterr(divide='ignore', invalid='ignore')
min_n_obs = 10


metadata_dict = utils.build_metadata_dict()
s_by_s, otu_labels, samples = utils.load_count_data()

param_env_dict = sine_parameter_utils.load_param_env_dict()
# get days
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])


idx_all = list(range(len(utils.env_variables_all)))
chunk_all = [idx_all[x:x+3] for x in range(0, len(idx_all), 3)]


fig = plt.figure(figsize = (20, 20))
fig.subplots_adjust(bottom= 0.15)


for chunk_idx, chunk in enumerate(chunk_all):

    for c_idx, c in enumerate(chunk):

        env_variable = utils.env_variables_all[c]
        
        ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

        env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])
        days_env = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

        to_keep_idx = ~numpy.isnan(env_variable_array)
        env_variable_array = env_variable_array[to_keep_idx]
        days_env = days_env[to_keep_idx]

        time_increments = list(range(1, len(days_env)-min_n_obs+1))
        delta_t_env = numpy.asarray([days_env[i] - days_env[0] for i in time_increments])
        autocorr_obs_env = [numpy.corrcoef(env_variable_array[i:], env_variable_array[:-i])[0,1] for i in time_increments]

        ax.scatter(delta_t_env, autocorr_obs_env, s=7, alpha=1, c='k')

        ax.set_xlabel('Time difference (days), ' + r'$\Delta t$', fontsize=8)
        #env_variable_label = utils.env_variable_no_unit_label_dict[env_variable][0].lower() + utils.env_variable_no_unit_label_dict[env_variable][1:]
        ax.set_ylabel('%s autocorrelation' % utils.env_variable_no_unit_label_dict[env_variable], fontsize=8)


        


fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%sautocorrelation_env.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

