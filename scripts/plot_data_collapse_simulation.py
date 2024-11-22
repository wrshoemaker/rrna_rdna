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

# numdifftools also installed
import pickle

import simulation_utils
import matplotlib.gridspec as gridspec



s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


param_dict = pickle.load(open(simulation_utils.data_collapse_simulation_path, "rb"))


fig = plt.figure(figsize = (6,5)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=2, ncols=1)

fig.text(0.2, 0.94, "Simulated gamma abundances\nwith constant carrying capacity", va='center', fontsize=16)

ax_data = fig.add_subplot(gs[0, :])
ax_data_rescaled = fig.add_subplot(gs[1, :])

ax_data.text(-0.1, 1.09, utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data.transAxes)
ax_data_rescaled.text(-0.1, 1.09, utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data_rescaled.transAxes)


ax_data_rescaled.xaxis.set_tick_params(labelsize=7)

ax_data.set_xlim([0, max(days)])
ax_data.set_xticks(minor_days, minor=True)
ax_data.set_xticks(major_days, minor=False)
ax_data.set_xticklabels(major_labels, minor=False, fontsize=7)
ax_data.yaxis.set_tick_params(labelsize=7)

ax_data_rescaled.xaxis.set_tick_params(labelsize=7)
ax_data_rescaled.yaxis.set_tick_params(labelsize=7)


ax_data.set_xlabel('Time (days)', fontsize=11)
ax_data_rescaled.set_xlabel('Rescaled time', fontsize=11)

ax_data.set_ylabel('CLR-transformed abund.', fontsize=10)
ax_data_rescaled.set_ylabel('Rescaled\nCLR-transformed abund.', fontsize=10)


rescaled_days_all = []
for afd_clr_otu_idx in range(len(param_dict['clr'])):

    clr = numpy.asarray(param_dict['clr'][afd_clr_otu_idx])
    amp = param_dict['amp_mle'][afd_clr_otu_idx]
    freq = param_dict['freq_mle'][afd_clr_otu_idx]
    phase = param_dict['phase_mle'][afd_clr_otu_idx]
    param_mean = param_dict['param_mean_mle'][afd_clr_otu_idx]

    rescaled_clr = (clr - numpy.log(param_mean))/amp
    rescaled_days = days*freq + phase

    
    if max(rescaled_days) > 100:
        continue 

    ax_data.plot(days, clr, lw=0.4, alpha=0.4, color='k')#, zorder=5-sine_param_combo_idx)
    ax_data.scatter(days, clr, s=1, alpha=0.4, color='k', zorder=2)

    ax_data_rescaled.plot(rescaled_days, rescaled_clr, lw=0.4, alpha=0.4, color='k')#, zorder=5-sine_param_combo_idx)    
    ax_data_rescaled.scatter(rescaled_days, rescaled_clr, s=1, alpha=0.4, color='k', zorder=2)

    rescaled_days_all.extend(rescaled_days.tolist())


rescaled_days_range = numpy.linspace(min(rescaled_days_all), max(rescaled_days_all), 1000)

ax_data_rescaled.plot(rescaled_days_range, numpy.sin(rescaled_days_range), lw=2, c='k', label='Sine function')
ax_data_rescaled.legend(loc='upper left', fontsize=6)


fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig_name = "%sdata_collapse_simulation.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


#print(param_dict.keys())