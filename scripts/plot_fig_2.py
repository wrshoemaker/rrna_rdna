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



method = 'mle'

s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()
param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
taxonomy_dict = utils.build_taxonomy_dict()

#print(param_dict['otu_labels'])

# first, large plot of the oscillations
focal_otu = param_dict['otu_labels'][0]
focal_otu_idx = param_dict['otu_labels'].index(focal_otu)

days_focal = param_dict['data']['days']['RNA'][focal_otu_idx]
afd_focal = param_dict['data']['clr_afd']['RNA'][focal_otu_idx]
amp_focal = param_dict['amp_%s' % method]['RNA'][focal_otu_idx]
freq_focal = param_dict['freq_%s' % method]['RNA'][focal_otu_idx]
phase_focal = param_dict['phase_%s' % method]['RNA'][focal_otu_idx]
param_mean_focal = param_dict['param_mean_%s' % method]['RNA'][focal_otu_idx]
beta_focal = param_dict['beta']['RNA'][focal_otu_idx]

sigma_focal = 2/(beta_focal+1)

days_range = numpy.linspace(min(days_focal), max(days_focal), 1000)
#model_prediction = amp_focal*numpy.sin(freq_focal*days_range+phase_focal) + numpy.log(param_mean_focal)
model_prediction = amp_focal*numpy.sin(freq_focal*days_range+phase_focal) + numpy.log(param_mean_focal) + numpy.log(1 - sigma_focal/2)

print( numpy.log(param_mean_focal) + numpy.log(1 - sigma_focal/2))

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(days_range, model_prediction, ls='-', lw=3, c=utils.dna_rna_color_dict['RNA'], zorder=1, label='Oscillating mean + gamma AFD')
ax.scatter(days_focal, afd_focal, s=8, alpha=1, c=utils.dna_rna_color_dict['RNA'], zorder=2)
ax.axhline(y= numpy.log(param_mean_focal) + numpy.log(1 - sigma_focal/2), ls=':', lw=3, zorder=3, c='k')#')
ax.set_xlabel("Time (days)", fontsize=14)
ax.set_ylabel("CLR-transformed abundance, " + utils.rescaled_label_clr_dict['RNA'], fontsize=14)

ax.set_xlim([0, max(days_focal)])
ax.set_xticks(minor_days, minor=True)
ax.set_xticks(major_days, minor=False)
ax.set_xticklabels(major_labels, minor=False, fontsize=7)
ax.yaxis.set_tick_params(labelsize=7)

ax.legend(loc='lower right', fontsize=10)
#ax.set_title('OTU 1', color='k', fontsize=14)
ax.set_title('ASV 1 (%s)' % taxonomy_dict[focal_otu]['family'], color='k', fontsize=14)

fig.subplots_adjust(hspace=0.35, wspace=0.25)
fig_name = "%sfig2_1.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



# second, 2x2 plot of rescaled and original data
fig = plt.figure(figsize = (8, 5)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=2, ncols=2)

# plot timeseries...
for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

    ax_data = fig.add_subplot(gs[data_type_idx, 0])
    ax_data_rescaled = fig.add_subplot(gs[data_type_idx, 1])

    #ax_data.text(-0.095, 1.06, utils.sub_plot_labels[data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data.transAxes)
    #ax_data_rescaled.text(-0.095, 1.06, utils.sub_plot_labels[8+data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data_rescaled.transAxes)


    for ax_ in [ax_data, ax_data_rescaled]:
        ax_.set_title(utils.rescaled_label_clr_dict[data_type], color=utils.dna_rna_color_dict[data_type], fontweight='bold', fontsize=14)
        ax_.yaxis.set_tick_params(labelsize=7)


    ax_data_rescaled.xaxis.set_tick_params(labelsize=7)
    ax_data.set_xlim([0, max(days)])
    ax_data.set_xticks(minor_days, minor=True)
    ax_data.set_xticks(major_days, minor=False)
    ax_data.set_xticklabels(major_labels, minor=False, fontsize=7)
    ax_data.yaxis.set_tick_params(labelsize=7)

    ax_data.set_xlabel('Time (days)', fontsize=11)
    ax_data_rescaled.set_xlabel('Rescaled time', fontsize=11)

    ax_data.set_ylabel('CLR-transformed abund.', fontsize=10)
    ax_data_rescaled.set_ylabel('Rescaled\nCLR-transformed abund.', fontsize=10)

    rescaled_days_all = []
    rescaled_clr_afd_all = []

    for otu_i_idx in range(len(param_dict['data']['days'][data_type])):

        days_i = numpy.asarray(param_dict['data']['days'][data_type][otu_i_idx])
        clr_afd_i = numpy.asarray(param_dict['data']['clr_afd'][data_type][otu_i_idx])
        
        ax_data.plot(days_i, clr_afd_i, lw=0.6, alpha=0.6, color=utils.dna_rna_color_dict[data_type])#, zorder=5-sine_param_combo_idx)
        ax_data.scatter(days_i, clr_afd_i, s=1, alpha=0.4, color=utils.dna_rna_color_dict[data_type], zorder=2)

        amp_mle_i = param_dict['amp_mle'][data_type][otu_i_idx]
        freq_mle_i = param_dict['freq_mle'][data_type][otu_i_idx]
        phase_mle_i = param_dict['phase_mle'][data_type][otu_i_idx]
        param_mean_mle_i = param_dict['param_mean_mle'][data_type][otu_i_idx]

        rescaled_days_i = days_i*freq_mle_i + phase_mle_i
        rescaled_clr_afd_i = (clr_afd_i - numpy.log(param_mean_mle_i))/amp_mle_i

        ax_data_rescaled.plot(rescaled_days_i, rescaled_clr_afd_i, lw=0.6, alpha=0.3, color=utils.dna_rna_color_dict[data_type])#, zorder=5-sine_param_combo_idx)
        ax_data_rescaled.scatter(rescaled_days_i, rescaled_clr_afd_i, s=1, alpha=0.4, color=utils.dna_rna_color_dict[data_type], zorder=2)

        rescaled_days_all.extend(rescaled_days_i.tolist())
        rescaled_clr_afd_all.extend(rescaled_clr_afd_i.tolist())


    # bin mean
    rescaled_days_all = numpy.asarray(rescaled_days_all)
    rescaled_clr_afd_all = numpy.asarray(rescaled_clr_afd_all)

    idx_to_keep = rescaled_days_all <= 50
    rescaled_days_all = rescaled_days_all[idx_to_keep]
    rescaled_clr_afd_all = rescaled_clr_afd_all[idx_to_keep]

    hist_all, bin_edges_all = numpy.histogram(rescaled_days_all, density=True, bins=30)
    bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_mean_all_to_keep = []
    bins_occupancies = []
    for i in range(0, len(bin_edges_all)-1 ):
        all_predicted_occupancies_log10_i = rescaled_clr_afd_all[ (rescaled_days_all>=bin_edges_all[i]) & (rescaled_days_all<bin_edges_all[i+1])]
        bins_mean_all_to_keep.append(bins_mean_all[i])
        bins_occupancies.append(numpy.mean(all_predicted_occupancies_log10_i))

    bins_mean_all_to_keep = numpy.asarray(bins_mean_all_to_keep)
    bins_occupancies = numpy.asarray(bins_occupancies)

    bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_occupancies))]
    bins_occupancies_no_nan = bins_occupancies[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_occupancies))]
    ax_data_rescaled.scatter(bins_mean_all_to_keep_no_nan, bins_occupancies_no_nan, s=17, marker="x", c='k', zorder=4, label='Mean over ASVs')

    rescaled_days_range = numpy.linspace(min(rescaled_days_all), max(rescaled_days_all), 1000)
    ax_data_rescaled.plot(rescaled_days_range, numpy.sin(rescaled_days_range), lw=1.5, c='k', label='Sine function (not a fit)', zorder=3)

    ax_data_rescaled.set_xlim([0, 50])

    #if data_type_idx == 0:
    ax_data_rescaled.legend(loc='lower center', fontsize=6)
    legend_elements = [Line2D([0], [0], marker='o', color=utils.dna_rna_color_dict[data_type], label='One ASV', markersize=5)]
    ax_data.legend(handles=legend_elements, loc='upper left', fontsize=6)



fig.subplots_adjust(hspace=0.45, wspace=0.35)
fig_name = "%sfig2_2.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

