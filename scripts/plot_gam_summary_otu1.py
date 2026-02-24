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


#gam_path = '%sgam_env_analysis_only_time.csv' % config.data_directory


#s_by_s, otu_labels, samples = utils.load_count_data()
#metadata_dict = utils.build_metadata_dict()
#sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
#env_var_dict = {}
#for env_variable_idx, env_variable in enumerate(utils.env_variable_all):
    
#    env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])
#    # remove nans
#    env_to_keep_idx = (~numpy.isnan(env_variable_array))
#    env_variable_array_clean = env_variable_array[env_to_keep_idx]

#    env_var_dict[env_variable] = numpy.std(env_variable_array_clean)



# get OTU1 data
#gam_dict = {}
#gam_file = open(gam_path, 'r')
#gam_header = gam_file.readline().strip().split(',')
#for line in gam_file:

#    if 'Otu000001' not in line:
#        continue

#    line = line.strip().split(',')
#    data_type = line[0].split('_', 1)[1]

#    stat = line[1]

#    for env_variable_idx in range(2, len(line)):

#        env_variable = gam_header[env_variable_idx]

#        if env_variable not in gam_dict:
#            gam_dict[env_variable] = {}

#        if data_type not in gam_dict[env_variable]:
#            gam_dict[env_variable][data_type] = {}
                
#        gam_dict[env_variable][data_type][stat] = float(line[env_variable_idx])

#        # rescale
#        if stat == 'coeff':
#            gam_dict[env_variable][data_type]['coeff_scaled'] = float(line[env_variable_idx]) * env_var_dict[env_variable]

            



#gam_file.close()

#env_variable_all = list(gam_dict.keys())


# add FDR correction
#for data_type in ['dna', 'rna', 'rna_dna']:

#    p_value_all = numpy.asarray([gam_dict[e][data_type]['p_value'] for e in env_variable_all])
#    p_value_all_corrected = fdrcorrection(p_value_all, alpha=0.05, method='indep', is_sorted=False)[1]

#    for env_variable_idx, env_variable in enumerate(env_variable_all):

#         gam_dict[env_variable][data_type]['p_value_fdr'] = p_value_all_corrected[env_variable_idx]




    #days_clean = days[env_to_keep_idx]

    #print(env_variable_array_clean)

#param_env_dict = pickle.load(open(config.data_directory + 'param_env_dict.pickle', "rb"))


gam_dict = utils.build_gam_coeff_dict()
env_variable_all = list(gam_dict.keys())


fig = plt.figure(figsize = (8.5, 4)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=1, ncols=2)

ax_coeff = fig.add_subplot(gs[0, 0])
ax_p_value = fig.add_subplot(gs[0, 1])

#y_axis_idx = numpy.asarray(range(len(env_variable_all)))[::-1]

#coeff_scaled_all = numpy.asarray([gam_dict[e]['dna']['coeff_scaled'] for e in env_variable_all])
#p_value_fdr_all = numpy.asarray([gam_dict[e][data_type]['p_value_fdr'] for e in env_variable_all])
env_variable_label_all = numpy.asarray([utils.env_variable_label_dict[e] for e in env_variable_all])
sort_idx = numpy.argsort(numpy.asarray([gam_dict[e]['dna']['coeff_scaled'] for e in env_variable_all]))[::-1]
env_variable_label_all = env_variable_label_all[sort_idx]


for data_type in ['dna', 'rna', 'rna_dna']:

    coeff_scaled_all = numpy.asarray([gam_dict[e][data_type]['coeff_scaled'] for e in env_variable_all])
    p_value_fdr_all = numpy.asarray([gam_dict[e][data_type]['p_value_fdr'] for e in env_variable_all])
    #sort_idx = numpy.argsort(coeff_scaled_all)

    coeff_scaled_all = coeff_scaled_all[sort_idx]
    p_value_fdr_all = p_value_fdr_all[sort_idx]
    p_value_fdr_all = -1*numpy.log10(p_value_fdr_all)

    ax_coeff.scatter(coeff_scaled_all, range(len(coeff_scaled_all)), alpha=0.7, s=30, color=utils.dna_rna_color_dict[data_type.upper()], label=utils.sample_label_dict[data_type.upper()])
    ax_p_value.scatter(p_value_fdr_all, range(len(coeff_scaled_all)), alpha=0.7, s=30, color=utils.dna_rna_color_dict[data_type.upper()], label=utils.sample_label_dict[data_type.upper()])


ax_coeff.set_xlabel('GAM coefficient * std. deviation', fontsize=10)
ax_coeff.set_yticks(range(len(coeff_scaled_all)))
ax_coeff.set_yticklabels(env_variable_label_all, fontsize=6, rotation=45)
ax_coeff.axvline(x=0, lw=2.5, ls=':', color='k', zorder=1)
ax_coeff.legend(loc='lower left', fontsize=6)

ax_p_value.set_xlabel(r'$- \mathrm{log}_{10}P$' + ', FDR-corrected', fontsize=10)
ax_p_value.set_yticks(range(len(coeff_scaled_all)))
ax_p_value.set_yticklabels(env_variable_label_all, fontsize=6, rotation=45)
ax_p_value.axvline(x=-1*numpy.log10(0.05), lw=2.5, ls=':', label=r'$P = 0.05$', color='k', zorder=1)
ax_p_value.legend(loc='lower right', fontsize=6)

fig.subplots_adjust(hspace=0.4, wspace=0.35)
fig_name = "%sgamma_summary_otu1.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()




#gam_env_analysis_only_time