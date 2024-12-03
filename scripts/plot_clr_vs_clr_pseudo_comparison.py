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


clr_color = '#87CEEB'
clr_all_otus_color = '#FFA500'

param_dict = pickle.load(open(simulation_utils.param_oscillation_artifact_simulation_path, "rb"))
param_all_otus_dict = pickle.load(open(simulation_utils.param_oscillation_artifact_simulation_clr_all_otus_path, "rb"))

s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


sigma_all = list(param_dict['true_abundance']['focal'].keys())
sigma_to_plot = sigma_all[0]
sine_param_combo_all = list(param_dict['true_abundance']['focal'][sigma_to_plot].keys())
sine_param_to_plot = sine_param_combo_all[-1]

fig = plt.figure(figsize = (8.5, 12))
fig.subplots_adjust(bottom= 0.15)

ax_sim_focal_abund = plt.subplot2grid((3, 2), (0, 0), colspan = 1)
ax_sim_nonfocal_abund = plt.subplot2grid((3, 2), (0, 1), colspan = 1)

ax_sim_focal_reads = plt.subplot2grid((3, 2), (1, 0), colspan = 1)
ax_sim_nonfocal_reads = plt.subplot2grid((3, 2), (1, 1), colspan = 1)
ax_sim_focal_reads_clr = ax_sim_focal_reads.twinx()
ax_sim_nonfocal_reads_clr = ax_sim_nonfocal_reads.twinx()

ax_sim_focal_param = plt.subplot2grid((3, 2), (2, 0), colspan = 1)
ax_sim_nonfocal_param = plt.subplot2grid((3, 2), (2, 1), colspan = 1)


# row 1
ax_sim_focal_abund.text(-0.1, 1.05, utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_focal_abund.transAxes)
ax_sim_nonfocal_abund.text(-0.1, 1.05, utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_nonfocal_abund.transAxes)

# row 2
ax_sim_focal_reads.text(-0.1, 1.05, utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_focal_reads.transAxes)
ax_sim_nonfocal_reads.text(-0.1, 1.05, utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_nonfocal_reads.transAxes)

# row 3
ax_sim_focal_param.text(-0.1, 1.05, utils.sub_plot_labels[4], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_focal_param.transAxes)
ax_sim_nonfocal_param.text(-0.1, 1.05, utils.sub_plot_labels[5], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_nonfocal_param.transAxes)



afd_true_focal = numpy.asarray(param_dict['true_abundance']['focal'][sigma_to_plot][sine_param_to_plot]['afd'])

mean_afd_true_focal = numpy.mean(afd_true_focal, axis=0)

afd_true_nonfocal = numpy.asarray(param_dict['true_abundance']['nonfocal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_true_nonfocal = numpy.mean(afd_true_nonfocal, axis=0)


ax_sim_focal_abund.plot(days, mean_afd_true_focal, lw=1, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)
ax_sim_focal_abund.scatter(days, mean_afd_true_focal, s=6, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)

ax_sim_nonfocal_abund.plot(days, mean_afd_true_nonfocal, lw=1, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_abund.scatter(days, mean_afd_true_nonfocal, s=6, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)


ax_sim_focal_abund.set_ylabel("True abundance", fontsize=12, color='k')
ax_sim_nonfocal_abund.set_ylabel("True abundance", fontsize=12, color='k')



# set y-lim for nonfocal
ax_sim_nonfocal_abund.set_ylim([0.6*min(mean_afd_true_nonfocal), (1/0.2)*max(mean_afd_true_nonfocal)])


ax_sim_focal_abund.set_title('Oscillating OTU', fontsize=12, fontweight='bold')
ax_sim_nonfocal_abund.set_title('Non-oscillating OTU', fontsize=12, fontweight='bold')


ax_sim_focal_abund.set_yscale('log', basey=10)
ax_sim_nonfocal_abund.set_yscale('log', basey=10)

ax_sim_focal_abund.set_yscale('log', basey=10)
ax_sim_nonfocal_abund.set_yscale('log', basey=10)




# format all axes
for ax_i in [ax_sim_focal_abund, ax_sim_nonfocal_abund, ax_sim_focal_abund, ax_sim_nonfocal_abund, ax_sim_focal_reads, ax_sim_nonfocal_reads]:

    ax_i.set_xlim([0, max(days)])
    ax_i.set_xticks(minor_days, minor=True)
    ax_i.set_xticks(major_days, minor=False)
    ax_i.set_xticklabels(major_labels, minor=False, fontsize=7)
    ax_i.set_xlabel("Time (days)", fontsize=12)
    ax_i.yaxis.set_tick_params(labelsize=7)


ax_sim_focal_reads_clr.tick_params(labelsize=7)
ax_sim_nonfocal_reads_clr.tick_params(labelsize=7)

ax_sim_focal_param.tick_params(labelsize=7)
ax_sim_nonfocal_param.tick_params(labelsize=7)


# X*10^0 are minor tick labels in matplotlib
ax_sim_focal_reads.yaxis.set_minor_formatter(ticker.NullFormatter())


# plot CLR

afd_clr_focal = numpy.asarray(param_dict['clr']['focal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_clr_focal = numpy.mean(afd_clr_focal, axis=0)

afd_clr_nonfocal = numpy.asarray(param_dict['clr']['nonfocal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_clr_nonfocal = numpy.mean(afd_clr_nonfocal, axis=0)


afd_clr_focal_all_otus = numpy.asarray(param_all_otus_dict['clr']['focal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_clr_focal_all_otus = numpy.mean(afd_clr_focal_all_otus, axis=0)


afd_clr_nonfocal_all_otus = numpy.asarray(param_all_otus_dict['clr']['nonfocal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_clr_nonfocal_all_otus = numpy.mean(afd_clr_nonfocal_all_otus, axis=0)


# focal timeseries
ax_sim_focal_reads.plot(days, mean_afd_clr_focal_all_otus, lw=1, alpha=1, color=clr_all_otus_color)
ax_sim_focal_reads.scatter(days, mean_afd_clr_focal_all_otus, s=6, alpha=1, color=clr_all_otus_color)

ax_sim_focal_reads_clr.plot(days, mean_afd_clr_focal, lw=1, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)
ax_sim_focal_reads_clr.scatter(days, mean_afd_clr_focal, s=6, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)



ax_sim_focal_reads.set_ylabel("CLR-transformed abund., pseudocount", fontsize=11, color=clr_all_otus_color, fontweight='bold')
ax_sim_focal_reads_clr.set_ylabel("CLR-transformed abundance", fontsize=11, color=clr_color, fontweight='bold')



# non-focal timeseries
ax_sim_nonfocal_reads.plot(days, mean_afd_clr_nonfocal_all_otus, lw=1, alpha=1, color=clr_all_otus_color)
ax_sim_nonfocal_reads.scatter(days, mean_afd_clr_nonfocal_all_otus, s=6, alpha=1, color=clr_all_otus_color)

ax_sim_nonfocal_reads_clr.plot(days, mean_afd_clr_nonfocal, lw=1, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_reads_clr.scatter(days, mean_afd_clr_nonfocal, s=6, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)


ax_sim_nonfocal_reads.set_ylabel("CLR-transformed abund., pseudocount", fontsize=11, color=clr_all_otus_color, fontweight='bold')
ax_sim_nonfocal_reads_clr.set_ylabel("CLR-transformed abundance", fontsize=11, color=clr_color, fontweight='bold')



# plot all predictions

true_amp = [sine_param_combo[0] for sine_param_combo in sine_param_combo_all]

clr_focal_amp = [numpy.mean(param_dict['clr']['focal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]
clr_focal_amp_all_otus = [numpy.mean(param_all_otus_dict['clr']['focal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]

clr_nonfocal_amp = [numpy.mean(param_dict['clr']['nonfocal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]
clr_nonfocal_amp_all_otus = [numpy.mean(param_all_otus_dict['clr']['nonfocal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]


min_amp = min(true_amp) - 0.1
max_amp = max(true_amp) + 0.1

ax_sim_focal_param.set_xlim([min_amp, max_amp])
ax_sim_focal_param.set_ylim([min_amp, max_amp])

ax_sim_nonfocal_param.set_xlim([min_amp, max_amp])
ax_sim_nonfocal_param.set_ylim([min_amp, max_amp])

ax_sim_focal_param.set_xticks(true_amp)
ax_sim_focal_param.set_xticklabels(true_amp)
ax_sim_focal_param.set_yticks(true_amp)
ax_sim_focal_param.set_yticklabels(true_amp)

ax_sim_nonfocal_param.set_xticks(true_amp)
ax_sim_nonfocal_param.set_xticklabels(true_amp)
ax_sim_nonfocal_param.set_yticks(true_amp)
ax_sim_nonfocal_param.set_yticklabels(true_amp)


ax_sim_focal_param.plot([min_amp, max_amp], [min_amp, max_amp], lw=2, ls=':', c='k', zorder=1, label='1:1')
ax_sim_nonfocal_param.plot([min_amp, max_amp], [min_amp, max_amp], lw=2, ls=':', c='k', zorder=1, label='1:1')

ax_sim_focal_param.plot(true_amp, clr_focal_amp, lw=2.5, alpha=1, color=clr_color, zorder=2)#, zorder=5-sine_param_combo_idx)
ax_sim_focal_param.scatter(true_amp, clr_focal_amp, s=40, alpha=1, color=clr_color, zorder=3, label='CLR-transformed abund.')#, zorder=5-sine_param_combo_idx)

ax_sim_focal_param.plot(true_amp, clr_focal_amp_all_otus, lw=2.5, alpha=1, color=clr_all_otus_color)#, zorder=5-sine_param_combo_idx)
ax_sim_focal_param.scatter(true_amp, clr_focal_amp_all_otus, s=40, alpha=1, color=clr_all_otus_color, label='CLR-transformed abund., pseudocount')#, zorder=5-sine_param_combo_idx)


ax_sim_nonfocal_param.plot(true_amp, clr_nonfocal_amp, lw=2.5, alpha=1, color=clr_color, zorder=2)#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_param.scatter(true_amp, clr_nonfocal_amp, s=40, alpha=1, color=clr_color, zorder=3, label='CLR-transformed abund.')#, zorder=5-sine_param_combo_idx)

ax_sim_nonfocal_param.plot(true_amp, clr_nonfocal_amp_all_otus, lw=2.5, alpha=1, color=clr_all_otus_color)#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_param.scatter(true_amp, clr_nonfocal_amp_all_otus, s=40, alpha=1, color=clr_all_otus_color, label='CLR-transformed abund., pseudocount')#, zorder=5-sine_param_combo_idx)

ax_sim_focal_param.set_xlabel("True amplitude of oscillating OTU" , fontsize=12)
ax_sim_focal_param.set_ylabel("Inferred amplitude\nof oscillating OTU", fontsize=11.5)

ax_sim_nonfocal_param.set_xlabel("True amplitude of oscillating OTU", fontsize=12)
ax_sim_nonfocal_param.set_ylabel("Inferred amplitude\nof non-oscillating OTU", fontsize=11.5)

ax_sim_focal_reads_clr.set_ylabel("CLR-transformed abundance", fontsize=11, color=clr_color, fontweight='bold')

ax_sim_nonfocal_param.axhline(y=0, ls='--', lw=2, zorder=0, c='k', label='True amp. of non-oscillating OTU')

ax_sim_focal_param.legend(loc="upper left", fontsize=8)
ax_sim_nonfocal_param.legend(loc="upper left", fontsize=8)







fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig_name = "%sclr_vs_clr_pseudo_comparison.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

