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

timeseries_col_n = 3
param_col_n = 2


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

metadata_dict = utils.build_metadata_dict()
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))

#fig = plt.figure(figsize = (8.5, 16))
#ax_data = plt.subplot2grid((6, 4), (0, 0))

fig = plt.figure(figsize = (10, 10)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=8, ncols=6)

# plot timeseries...
for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

    data_col_left = data_type_idx*timeseries_col_n
    data_col_right = data_type_idx*timeseries_col_n + timeseries_col_n

    ax_data = fig.add_subplot(gs[0:2, data_col_left:data_col_right])
    ax_data_rescaled = fig.add_subplot(gs[6:, data_col_left:data_col_right])

    ax_data.text(-0.095, 1.06, utils.sub_plot_labels[data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data.transAxes)
    ax_data_rescaled.text(-0.095, 1.06, utils.sub_plot_labels[8+data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_data_rescaled.transAxes)


    for ax_ in [ax_data, ax_data_rescaled]:
        ax_.set_title(data_type, color=utils.dna_rna_color_dict[data_type], fontweight='bold', fontsize=14)
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

        ax_data_rescaled.plot(rescaled_days_i, rescaled_clr_afd_i, lw=0.6, alpha=0.6, color=utils.dna_rna_color_dict[data_type])#, zorder=5-sine_param_combo_idx)
        ax_data_rescaled.scatter(rescaled_days_i, rescaled_clr_afd_i, s=1, alpha=0.4, color=utils.dna_rna_color_dict[data_type], zorder=2)

        rescaled_days_all.extend(rescaled_days_i.tolist())


    rescaled_days_range = numpy.linspace(min(rescaled_days_all), max(rescaled_days_all), 1000)
    ax_data_rescaled.plot(rescaled_days_range, numpy.sin(rescaled_days_range), lw=2, c='k', label='Sine function')

    if data_type_idx == 0:
        ax_data_rescaled.legend(loc='upper left', fontsize=6)


# plot parameters 
for param_idx, param in enumerate(['amp', 'freq', 'phase']):

    param_col_left = param_idx*param_col_n
    param_col_right = param_idx*param_col_n + param_col_n

    param_dna = numpy.asarray(param_dict['%s_mle' % param]['DNA'])
    param_rna = numpy.asarray(param_dict['%s_mle' % param]['RNA'])

    ax_param_dist = fig.add_subplot(gs[2:4, param_col_left:param_col_right])
    ax_param_scatter = fig.add_subplot(gs[4:6, param_col_left:param_col_right])

    ax_param_dist.text(-0.1, 1.07, utils.sub_plot_labels[2+param_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_param_dist.transAxes)
    ax_param_scatter.text(-0.1, 1.07, utils.sub_plot_labels[5+param_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_param_scatter.transAxes)


    if param == 'freq':
        param_dna = 2*numpy.pi / param_dna
        param_rna = 2*numpy.pi / param_rna

        ax_param_dist.axvline(x=365, ls='--', color='k', lw=2, label='Yearly oscillations', zorder=3)
        ax_param_dist.axvline(x=365/2, ls=':', color='k', lw=2, label='Semi-yearly oscillations', zorder=3)


    ax_param_dist.xaxis.set_tick_params(labelsize=7)
    ax_param_dist.yaxis.set_tick_params(labelsize=7)

    ax_param_scatter.xaxis.set_tick_params(labelsize=7)
    ax_param_scatter.yaxis.set_tick_params(labelsize=7)

    axis_label = sine_parameter_utils.param_label_dict[param]
    ax_param_dist.set_xlabel('%s, %s' % (axis_label, sine_parameter_utils.param_label_dict_latex[param]), fontsize=10)
    ax_param_dist.set_ylabel('Probability density', fontsize=10)

    ax_param_scatter.set_xlabel('%s, DNA' % sine_parameter_utils.param_label_no_days_dict[param], fontsize=10)
    ax_param_scatter.set_ylabel('%s, RNA' % sine_parameter_utils.param_label_no_days_dict[param], fontsize=10)

    ax_param_dist.hist(param_dna, 8, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['DNA'], label='DNA')
    ax_param_dist.hist(param_rna, 8, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['RNA'], label='RNA')

    #ax_param_dist.set_xlabel(param_label_dict[param_label], fontsize=11)
    ax_param_scatter.scatter(param_dna, param_rna, s=6, color='k', alpha=1, zorder=2)
    param_concat = numpy.concatenate([param_dna, param_rna])
    min_param = min(param_concat) * 0.8
    max_param = max(param_concat) * 1.2
    ax_param_scatter.set_xlim([min_param, max_param])
    ax_param_scatter.set_ylim([min_param, max_param])
    ax_param_scatter.plot([min_param, max_param], [min_param, max_param], lw=2, ls=':', c='k', zorder=1, label='1:1')

    rho = numpy.corrcoef(param_dna, param_rna)[0,1]
    ax_param_scatter.text(0.24, 0.7, r'$\rho^{2} = $' + str(round(rho**2, 3)), fontsize=10, ha='center', va='center', transform=ax_param_scatter.transAxes)
            
    if param_idx == 0:
        ax_param_dist.legend(loc='upper right', fontsize=6)
        ax_param_scatter.legend(loc='upper left', fontsize=6)


    if param == 'phase':
        phase_ticks = [0, numpy.pi/2, numpy.pi, 3*numpy.pi/2, 2*numpy.pi]
        phase_tick_labels = [r'0', r'$\frac{\pi}{2}$',  r'$\pi$',  r'$\frac{3\pi}{2}$', r'$2\pi$',]
        ax_param_dist.set_xticks(phase_ticks)
        ax_param_dist.set_xticklabels(phase_tick_labels)
        ax_param_dist.xaxis.set_tick_params(labelsize=7)

        ax_param_scatter.set_xticks(phase_ticks)
        ax_param_scatter.set_xticklabels(phase_tick_labels)
        ax_param_scatter.xaxis.set_tick_params(labelsize=7)

        ax_param_scatter.set_yticks(phase_ticks)
        ax_param_scatter.set_yticklabels(phase_tick_labels)
        ax_param_scatter.yaxis.set_tick_params(labelsize=7)

        ax_param_scatter.set_xlim([0, 2*numpy.pi])
        ax_param_scatter.set_ylim([0, 2*numpy.pi])


    if param == 'freq':

        legend_elements = [Line2D([0], [0], color='k', ls='--', lw=2, label='Yearly'),
                            Line2D([0], [0], color='k', ls=':', lw=2, label='Bi-yearly')]
        ax_param_dist.legend(handles=legend_elements, fontsize=6, loc='upper right')
        


    #param_t = numpy.asarray(param_t)
    #ax_compare.scatter(param_t[param_reference_idx], list(range(len(param_t))), s=6, color=utils.dna_rna_color_dict[t], zorder=2)
    #ax_compare.plot(param_t, list(range(len(param_t))), lw=1, alpha=0.6, color=utils.dna_rna_color_dict[t], zorder=1)


    # lists are already indexed
    #ax_param_scatter = fig.add_subplot(gs[4:6, param_col_left:param_col_right])



fig.subplots_adjust(hspace=1.8, wspace=1)
fig_name = "%sfig4.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()