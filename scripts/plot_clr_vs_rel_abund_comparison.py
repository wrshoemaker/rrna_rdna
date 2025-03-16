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


# get empirical data
s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


#print(s_by_s.shape)


s = 3
n_sites = len(days)
S = 1000
N = 100000

# run simulation
#simulation_utils.oscillation_artifact_simulation(0.001, s, S, N, 'exp', [0.1, 0.3, 0.5], n_sites, focal_amp_all=[0, 0.5, 1, 1.5, 2], n_iter=10)
# load simulation results
param_dict = pickle.load(open(simulation_utils.param_oscillation_artifact_simulation_path, "rb"))

sigma_all = list(param_dict['true_abundance']['focal'].keys())
sigma_to_plot = sigma_all[0]
sine_param_combo_all = list(param_dict['true_abundance']['focal'][sigma_to_plot].keys())
sine_param_to_plot = sine_param_combo_all[-1]
sorted(sine_param_combo_all, key=itemgetter(0))

clr_colormap = utils.make_colormap('DNA', len(sine_param_combo_all), lower_linspace_bound=0.2)
rel_colormap = utils.make_colormap('RNA', len(sine_param_combo_all), lower_linspace_bound=0.2)

# offset so you can see lighter color.
true_abund_colormap = cm.get_cmap('Greys')(numpy.linspace(0.3, 1.0, len(sine_param_combo_all)) )

sine_to_plot_idx = [0, 2, 4]

metadata_dict = utils.build_metadata_dict()
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


rel_color = utils.transformation_color_dict['rel']
clr_color = utils.transformation_color_dict['clr']

# set up plot...
fig = plt.figure(figsize = (8.5, 16))
fig.subplots_adjust(bottom= 0.15)

ax_data = plt.subplot2grid((4, 2), (0, 0))
ax_data_clr = ax_data.twinx()

#ax_model = plt.subplot2grid((4, 2), (0, 1))

#ax_sim_abund = plt.subplot2grid((4, 2), (1, 0), colspan = 2)

ax_sim_focal_abund = plt.subplot2grid((4, 2), (1, 0), colspan = 1)
ax_sim_nonfocal_abund = plt.subplot2grid((4, 2), (1, 1), colspan = 1)

ax_sim_focal_reads = plt.subplot2grid((4, 2), (2, 0), colspan = 1)
ax_sim_nonfocal_reads = plt.subplot2grid((4, 2), (2, 1), colspan = 1)
ax_sim_focal_reads_clr = ax_sim_focal_reads.twinx()
ax_sim_nonfocal_reads_clr = ax_sim_nonfocal_reads.twinx()

ax_sim_focal_param = plt.subplot2grid((4, 2), (3, 0), colspan = 1)
ax_sim_nonfocal_param = plt.subplot2grid((4, 2), (3, 1), colspan = 1)

# row 1
ax_data.text(-0.13, 1.04, utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data.transAxes)
ax_data.text(1.4, 1.04, utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data.transAxes)

# row 2
ax_sim_focal_abund.text(-0.1, 1.04, utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_focal_abund.transAxes)
ax_sim_nonfocal_abund.text(-0.1, 1.04, utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_nonfocal_abund.transAxes)

ax_sim_focal_reads.text(-0.1, 1.04, utils.sub_plot_labels[4], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_focal_reads.transAxes)
ax_sim_nonfocal_reads.text(-0.1, 1.04, utils.sub_plot_labels[5], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_nonfocal_reads.transAxes)
#ax_sim_nonfocal_reads_clr.text(-0.1, 1.04, utils.sub_plot_labels[5], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_nonfocal_reads_clr.transAxes)

ax_sim_focal_param.text(-0.1, 1.04, utils.sub_plot_labels[6], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_focal_param.transAxes)
ax_sim_nonfocal_param.text(-0.1, 1.04, utils.sub_plot_labels[7], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_sim_nonfocal_param.transAxes)




# plot motivation from data.
s_by_s_rna = s_by_s[:,(sample_type=='RNA')]
n_reads = numpy.sum(s_by_s_rna, axis=0)

focal_otu_idx = numpy.where(otu_labels=='Otu000001')[0][0]
focal_otu_afd = s_by_s_rna[focal_otu_idx,:]

focal_otu_afd_rel = focal_otu_afd/n_reads
#focal_otu_afd_clr = numpy.log(focal_otu_afd/stats.gmean(focal_otu_afd))

clr_s_by_s_dna, clr_s_by_s_rna, otu_labels_occupancy = utils.clr_transform_subset(s_by_s, otu_labels, samples)
focal_otu_afd_clr = clr_s_by_s_dna[0,:]


ax_data.plot(days, focal_otu_afd_rel, lw=1, alpha=1, color=rel_color, zorder=1)
ax_data.scatter(days, focal_otu_afd_rel, s=6, alpha=1, color=rel_color, zorder=1)
ax_data_clr.plot(days, focal_otu_afd_clr, lw=1, alpha=1, color=clr_color, zorder=2)
ax_data_clr.scatter(days, focal_otu_afd_clr, s=8, alpha=1, color=clr_color, zorder=2)


#ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict[data_type])
ax_data.set_ylabel("Relative abundance", fontsize=11, color=rel_color, fontweight='bold')
ax_data_clr.set_ylabel("CLR-transformed abundance", fontsize=11, color=clr_color, fontweight='bold')
ax_data.axhline(y=1, lw=2.5, ls=':', label='Max. relative abundance', color=rel_color)
ax_data.set_title('Observed RNA abundance of OTU 1', fontsize=12, fontweight='bold')
ax_data.set_yscale('log', basey=10)

ax_data.legend(loc="lower left", fontsize=8)




# plot simulated trajectories of *true* abundance
# plot simulated trajectories of *sampled* abundance

#mean_afd_true_nonfocal_all = []
#for sine_param_combo_idx, sine_param_combo in enumerate(sine_param_combo_all):

#if sine_param_combo_idx == 1
#if sine_param_combo_idx not in sine_to_plot_idx:
#    continue

#afd_true_focal = numpy.asarray(param_dict['true_abundance']['focal'][sigma_to_plot][sine_param_combo]['afd'])
afd_true_focal = numpy.asarray(param_dict['true_abundance']['focal'][sigma_to_plot][sine_param_to_plot]['afd'])

mean_afd_true_focal = numpy.mean(afd_true_focal, axis=0)

afd_true_nonfocal = numpy.asarray(param_dict['true_abundance']['nonfocal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_true_nonfocal = numpy.mean(afd_true_nonfocal, axis=0)


ax_sim_focal_abund.plot(days, mean_afd_true_focal, lw=1, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)
ax_sim_focal_abund.scatter(days, mean_afd_true_focal, s=6, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)

ax_sim_nonfocal_abund.plot(days, mean_afd_true_nonfocal, lw=1, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_abund.scatter(days, mean_afd_true_nonfocal, s=6, alpha=1, color='k')#, zorder=5-sine_param_combo_idx)

#mean_afd_true_nonfocal_all.append(mean_afd_true_nonfocal)


# sampled
afd_logrel_focal = numpy.asarray(param_dict['log_rel']['focal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_logrel_focal = numpy.mean(afd_logrel_focal, axis=0)

afd_logrel_nonfocal = numpy.asarray(param_dict['log_rel']['nonfocal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_logrel_nonfocal = numpy.mean(afd_logrel_nonfocal, axis=0)

afd_clr_focal = numpy.asarray(param_dict['clr']['focal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_clr_focal = numpy.mean(afd_clr_focal, axis=0)

afd_clr_nonfocal = numpy.asarray(param_dict['clr']['nonfocal'][sigma_to_plot][sine_param_to_plot]['afd'])
mean_afd_clr_nonfocal = numpy.mean(afd_clr_nonfocal, axis=0)

# log relative
ax_sim_focal_reads.plot(days, 10**mean_afd_logrel_focal, lw=1, alpha=1, color=rel_color)#, zorder=5-sine_param_combo_idx)
ax_sim_focal_reads.scatter(days, 10**mean_afd_logrel_focal, s=6, alpha=1, color=rel_color)#, zorder=5-sine_param_combo_idx)

ax_sim_nonfocal_reads.plot(days, 10**mean_afd_logrel_nonfocal, lw=1, alpha=1, color=rel_color)#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_reads.scatter(days, 10**mean_afd_logrel_nonfocal, s=6, alpha=1, color=rel_color)#, zorder=5-sine_param_combo_idx)


# CLR
ax_sim_focal_reads_clr.plot(days, mean_afd_clr_focal, lw=1, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)
ax_sim_focal_reads_clr.scatter(days, mean_afd_clr_focal, s=6, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)

ax_sim_nonfocal_reads_clr.plot(days, mean_afd_clr_nonfocal, lw=1, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_reads_clr.scatter(days, mean_afd_clr_nonfocal, s=6, alpha=1, color=clr_color)#, zorder=5-sine_param_combo_idx)





ax_sim_focal_abund.set_ylabel("True abundance", fontsize=12, color='k')
ax_sim_nonfocal_abund.set_ylabel("True abundance", fontsize=12, color='k')



# set y-lim for nonfocal
#mean_afd_true_nonfocal_all_flat = numpy.concatenate(mean_afd_true_nonfocal_all)
ax_sim_nonfocal_abund.set_ylim([0.6*min(mean_afd_true_nonfocal), (1/0.2)*max(mean_afd_true_nonfocal)])


ax_sim_focal_abund.set_title('Oscillating OTU', fontsize=12, fontweight='bold')
ax_sim_nonfocal_abund.set_title('Non-oscillating OTU', fontsize=12, fontweight='bold')




ax_sim_focal_reads.set_ylabel("Relative abundance", fontsize=11, color=rel_color, fontweight='bold')
ax_sim_focal_reads_clr.set_ylabel("CLR-transformed abundance", fontsize=11, color=clr_color, fontweight='bold')

ax_sim_nonfocal_reads.set_ylabel("Relative abundance", fontsize=11, color=rel_color, fontweight='bold')
ax_sim_nonfocal_reads_clr.set_ylabel("CLR-transformed abundance", fontsize=11, color=clr_color, fontweight='bold')


ax_sim_focal_reads.axhline(y=1, lw=2.5, ls=':', label='Upper bound of rel. abund.', color=rel_color)
#ax_sim_focal_reads.set_ylim([0.2*min(10**mean_afd_logrel_focal), 1.1])

ax_sim_focal_reads.set_title('Oscillating OTU + sampling', fontsize=12, fontweight='bold')
ax_sim_nonfocal_reads.set_title('Non-oscillating OTU + sampling', fontsize=12, fontweight='bold')

ax_sim_focal_abund.set_yscale('log', basey=10)
ax_sim_nonfocal_abund.set_yscale('log', basey=10)

ax_sim_focal_abund.set_yscale('log', basey=10)
ax_sim_nonfocal_abund.set_yscale('log', basey=10)

#ax_sim_focal_reads_labels = [item.get_text() for item in ax_sim_focal_reads.get_yticklabels()]



ax_sim_focal_reads.set_yscale('log', basey=10)
ax_sim_nonfocal_reads.set_yscale('log', basey=10)


for ax_i in [ax_data, ax_sim_focal_abund, ax_sim_nonfocal_abund, ax_sim_focal_abund, ax_sim_nonfocal_abund, ax_sim_focal_reads, ax_sim_nonfocal_reads]:

    ax_i.set_xlim([0, max(days)])
    ax_i.set_xticks(minor_days, minor=True)
    ax_i.set_xticks(major_days, minor=False)
    ax_i.set_xticklabels(major_labels, minor=False, fontsize=7)
    ax_i.set_xlabel("Time (days)", fontsize=12)
    ax_i.yaxis.set_tick_params(labelsize=7)


ax_data_clr.yaxis.set_tick_params(labelsize=7)

ax_sim_focal_reads_clr.tick_params(labelsize=7)
ax_sim_nonfocal_reads_clr.tick_params(labelsize=7)

ax_sim_focal_param.tick_params(labelsize=7)
ax_sim_nonfocal_param.tick_params(labelsize=7)


# X*10^0 are minor tick labels in matplotlib
ax_sim_focal_reads.yaxis.set_minor_formatter(ticker.NullFormatter())


# true vs. inferred parameters... 
true_amp = [sine_param_combo[0] for sine_param_combo in sine_param_combo_all]

clr_focal_amp = [numpy.mean(param_dict['clr']['focal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]
log_rel_focal_amp = [numpy.mean(param_dict['log_rel']['focal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]

clr_nonfocal_amp = [numpy.mean(param_dict['clr']['nonfocal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]
log_rel_nonfocal_amp = [numpy.mean(param_dict['log_rel']['nonfocal'][sigma_to_plot][sine_param_combo]['amp_mle']) for sine_param_combo in sine_param_combo_all]


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

ax_sim_focal_param.plot(true_amp, log_rel_focal_amp, lw=2.5, alpha=1, color=rel_color)#, zorder=5-sine_param_combo_idx)
ax_sim_focal_param.scatter(true_amp, log_rel_focal_amp, s=40, alpha=1, color=rel_color, label='Log relative abund.')#, zorder=5-sine_param_combo_idx)


ax_sim_nonfocal_param.plot(true_amp, clr_nonfocal_amp, lw=2.5, alpha=1, color=clr_color, zorder=2)#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_param.scatter(true_amp, clr_nonfocal_amp, s=40, alpha=1, color=clr_color, zorder=3, label='CLR-transformed abund.')#, zorder=5-sine_param_combo_idx)

ax_sim_nonfocal_param.plot(true_amp, log_rel_nonfocal_amp, lw=2.5, alpha=1, color=rel_color)#, zorder=5-sine_param_combo_idx)
ax_sim_nonfocal_param.scatter(true_amp, log_rel_nonfocal_amp, s=40, alpha=1, color=rel_color, label='Log relative abund.')#, zorder=5-sine_param_combo_idx)

#ax_sim_focal_param.set_title('Oscillating OTU + sampling', fontsize=12, fontweight='bold')
#ax_sim_nonfocal_param.set_title('Non-oscillating OTU + sampling', fontsize=12, fontweight='bold')

ax_sim_focal_param.set_xlabel("True amplitude of oscillating OTU" , fontsize=12)
ax_sim_focal_param.set_ylabel("Inferred amplitude of oscillating OTU", fontsize=11.5)

ax_sim_nonfocal_param.set_xlabel("True amplitude of oscillating OTU", fontsize=12)
ax_sim_nonfocal_param.set_ylabel("Inferred amplitude of non-oscillating OTU", fontsize=11.5)

#ax_sim_focal_reads_clr.set_ylabel("CLR-transformed abundance", fontsize=11, color=clr_color, fontweight='bold')

ax_sim_nonfocal_param.axhline(y=0, ls='--', lw=2, zorder=0, c='k', label='True amp. of non-oscillating OTU')

ax_sim_focal_param.legend(loc="upper left", fontsize=8)
ax_sim_nonfocal_param.legend(loc="upper left", fontsize=8)






fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig_name = "%sclr_vs_rel_abund_comparison.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


