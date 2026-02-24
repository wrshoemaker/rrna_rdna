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

import sine_parameter_utils

# numdifftools also installed
import pickle
from scipy.stats import gamma, loggamma, nbinom, norm


import plot_predict_change_dna

#gam_path = '%sgam_env_analysis_only_time.csv' % config.data_directory


gam_dict = utils.load_gam()
env_variable_all = numpy.asarray(list(gam_dict.keys()))

#print(gam_dict[otu]['dna']['total_phosphorus'].keys())
env_variable_label_all = numpy.asarray([utils.env_variable_no_unit_label_dict[e] for e in env_variable_all])
sort_idx = numpy.argsort(numpy.asarray([gam_dict[e]['dna']['coeff_scaled'] for e in env_variable_all]))[::-1]
env_variable_label_all = env_variable_label_all[sort_idx]
env_variable_all = env_variable_all[sort_idx]

#print(env_variable_all)


fig = plt.figure(figsize = (4, 8))
fig.subplots_adjust(bottom= 0.15)
ax = plt.subplot2grid((1, 1), (0, 0))


for data_type in ['dna', 'rna']:

    coeff_scaled_all = numpy.asarray([gam_dict[e][data_type]['coeff_scaled'] for e in env_variable_all])
    p_value_fdr_all = numpy.asarray([gam_dict[e][data_type]['p_value_fdr'] for e in env_variable_all])

    c = utils.dna_rna_color_dict[data_type.upper()]

    for idx_ in range(len(p_value_fdr_all)):

        if p_value_fdr_all[idx_] < 0.05:
            facecolor = c
        else:
            facecolor = 'none'

        ax.scatter(coeff_scaled_all[idx_], idx_, alpha=1, s=310, linewidth=3, edgecolor=c, facecolor=facecolor, zorder=2)




ax.axvline(x=0, lw=3.5, ls=':', color='k', zorder=1)

#for env in utils.env_variable_to_plot:

#env_variable_array = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples[(sample_type=='RNA')]])


ax.set_xlabel('Standardized GAM coefficient', fontsize=14)
ax.set_yticks(range(len(coeff_scaled_all)))
ax.set_yticklabels(env_variable_label_all, fontsize=12, rotation=45)

legend_elements = [
    Line2D([0], [0],
           marker='o',
           color='black',
           linewidth=5,
           markerfacecolor='black',   # closed circle
           markersize=12,
           linestyle='None',
           label=r'$P_{\mathrm{FDR}} < 0.05$'),

    Line2D([0], [0],
           marker='o',
           color='black',
           linewidth=5,
           markerfacecolor='none',    # open circle
           markeredgecolor='black',
           markersize=12,
           linestyle='None',
           label=r'$P_{\mathrm{FDR}} \nless 0.05$')
]

ax.legend(handles=legend_elements, loc='lower left')


#print(metadata_dict)


#def plot_env_slope():



fig.subplots_adjust(hspace=0.4, wspace=0.35)
fig_name = "%sgam_slope_all_env_fig4.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
