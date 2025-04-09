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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


import sine_parameter_utils

# numdifftools also installed
import pickle

import simulation_utils


method = 'mle'

s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

metadata_dict = utils.build_metadata_dict()
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))




fig = plt.figure(figsize = (12.5, 4)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=1, ncols=3)


ax_timescale = fig.add_subplot(gs[0, 0])
ax_amp = fig.add_subplot(gs[0, 2])
ax_phase = fig.add_subplot(gs[0, 1])


for ax_idx, ax_ in enumerate([ax_timescale, ax_phase, ax_amp]):

    ax_.set_ylabel('Probability density', fontsize=12)
    ax_.text(-0.095, 1.06, utils.sub_plot_labels[ax_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_.transAxes)
    #ax_amp.text(-0.095, 1.06, utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_amp.transAxes)
    #ax_phase.text(-0.095, 1.06, utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_phase.transAxes)


# format data
freq_dna = numpy.asarray(param_dict['freq_mle']['DNA'])
freq_rna = numpy.asarray(param_dict['freq_mle']['RNA'])
timescale_dna = 2*numpy.pi/freq_dna
timescale_rna = 2*numpy.pi/freq_rna

amp_dna = numpy.asarray(param_dict['amp_mle']['DNA'])
amp_rna = numpy.asarray(param_dict['amp_mle']['RNA'])

phase_dna = numpy.asarray(param_dict['phase_mle']['DNA'])
phase_rna = numpy.asarray(param_dict['phase_mle']['RNA'])

delta_phase = phase_rna - phase_dna
# max delta can be +/- pi
delta_phase_new = []
for d in delta_phase:

    if d > numpy.pi:
        delta_phase_new.append(d - 2*numpy.pi)
    elif d < -numpy.pi:
        delta_phase_new.append(d + 2*numpy.pi)
    else:
        delta_phase_new.append(d)


delta_phase_new = numpy.asarray(delta_phase_new)


###
# plot oscillation timescale
###

ax_timescale.hist(timescale_dna, 8, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['DNA'], label='DNA')
ax_timescale.hist(timescale_rna, 8, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['RNA'], label='RNA')
ax_timescale.set_xlabel('Oscillation timescale (days), ' + r'$\tau_{i}^{\mathrm{env}}$', fontsize=12, zorder=3)

ax_timescale.axvline(x=365, ls='--', color='k', lw=2, label='Yearly', zorder=3)
#ax_timescale.axvline(x=365/2, ymin=0, ymax=0.0037, ls=':', color='k', lw=2, label='Semi-yearly', zorder=3)
ax_timescale.legend(loc='lower left', fontsize=6)
 
ax_timescale.xaxis.set_tick_params(labelsize=7)
ax_timescale.yaxis.set_tick_params(labelsize=7)


#ax_timescale.add_patch(Rectangle((0.12, 0.54), 0.28, 0.28, alpha=1, facecolor='w'))

 

#ax_timescale_rho = inset_axes(ax_timescale, width="100%", height="100%", bbox_to_anchor=(0.72,0.64,0.28,0.28), bbox_transform=ax_timescale.transAxes, loc='upper right')
ax_timescale_rho = inset_axes(ax_timescale, width="100%", height="100%", bbox_to_anchor=(0.12,0.64,0.28,0.28), bbox_transform=ax_timescale.transAxes, loc='upper right')
ax_timescale_rho.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
ax_timescale_rho.xaxis.set_tick_params(labelsize=6)



ax_timescale_rho.scatter(timescale_dna, timescale_rna, s=6, color='k', alpha=0.7, zorder=2)
param_concat = numpy.concatenate([timescale_dna, timescale_rna])
min_param = min(param_concat) * 0.8
max_param = max(param_concat) * 1.2
ax_timescale_rho.set_xlim([min_param, max_param])
ax_timescale_rho.set_ylim([min_param, max_param])
ax_timescale_rho.plot([min_param, max_param], [min_param, max_param], lw=2, ls=':', c='k', zorder=1, label='1:1')

ax_timescale_rho.set_xlabel(r'$\tau_{i}^{\mathrm{env}}$' + ', DNA', fontsize=8)
ax_timescale_rho.set_ylabel(r'$\tau_{i}^{\mathrm{env}}$' + ', RNA', fontsize=8)

ax_timescale_rho.legend(loc='upper left', fontsize=5)

timescale_rho = numpy.corrcoef(timescale_dna, timescale_rna)[0,1]
#ax_param_scatter.text(0.24, 0.7, r'$\rho^{2} = $' + str(round(rho**2, 3)), fontsize=10, ha='center', va='center', transform=ax_param_scatter.transAxes)
ax_timescale_rho.set_title(r'$\rho^{2} = $' + str(round(timescale_rho**2, 3)), fontsize=8)



###
# amplitude
###

ax_amp.hist(amp_dna, 8, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['DNA'], label='DNA')
ax_amp.hist(amp_rna, 8, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['RNA'], label='RNA')
ax_amp.set_xlabel('Amplitude, ' + r'$A_{i}$', fontsize=12, zorder=3)


ax_amp.xaxis.set_tick_params(labelsize=7)
ax_amp.yaxis.set_tick_params(labelsize=7)



ax_amp_rho = inset_axes(ax_amp, width="100%", height="100%", bbox_to_anchor=(0.72,0.64,0.28,0.28), bbox_transform=ax_amp.transAxes, loc='upper right')
ax_amp_rho.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
ax_amp_rho.xaxis.set_tick_params(labelsize=6)

ax_amp_rho.scatter(amp_dna, amp_rna, s=6, color='k', alpha=0.7, zorder=2)
param_concat = numpy.concatenate([amp_dna, amp_rna])
min_param = min(param_concat) * 0.8
max_param = max(param_concat) * 1.2
ax_amp_rho.set_xlim([min_param, max_param])
ax_amp_rho.set_ylim([min_param, max_param])
ax_amp_rho.plot([min_param, max_param], [min_param, max_param], lw=2, ls=':', c='k', zorder=1, label='1:1')

ax_amp_rho.set_xlabel(r'$A_{i}$' + ', DNA', fontsize=8)
ax_amp_rho.set_ylabel(r'$A_{i}$' + ', RNA', fontsize=8)

#ax_amp_rho.legend(loc='upper left', fontsize=5)

amp_rho = numpy.corrcoef(amp_dna, amp_rna)[0,1]
ax_amp_rho.set_title(r'$\rho^{2} = $' + str(round(amp_rho**2, 3)), fontsize=8)




###
# plot phase
###
ax_phase.hist(delta_phase_new, 8, histtype='step', density=True, stacked=True, lw=2, fill=False, color='k')
ax_phase.set_xlabel('Phase difference, ' + r'$\Delta \psi_{i} = \psi_{i}^{\mathrm{RNA}} -\psi_{i}^{\mathrm{DNA}}$', fontsize=11, zorder=3)
#ax_phase.axvline(x=0, ls=':', color='k', lw=3, label=r'$\Delta \psi_{i}=0$', zorder=2)
ax_phase.axvline(x=numpy.mean(delta_phase_new), ls=':', color='k', lw=3, label='Mean ' + r'$\Delta \psi_{i}$' + ' = ' + str(round(numpy.mean(delta_phase_new), 3)), zorder=2)


phase_ticks = [-numpy.pi, -0.5*numpy.pi, 0, 0.5*numpy.pi, numpy.pi]
phase_tick_labels = [r'$-\pi$', r'$-\frac{\pi}{2}$', r'0', r'$\frac{\pi}{2}$', r'$\pi$']
ax_phase.set_xticks(phase_ticks)
ax_phase.set_xticklabels(phase_tick_labels)
ax_phase.xaxis.set_tick_params(labelsize=7)
ax_phase.yaxis.set_tick_params(labelsize=7)

y_max = 0.7
ax_phase.set_ylim([0,y_max])
ax_phase.set_xlim([-numpy.pi,numpy.pi])



# fill in beetween
phase_range = numpy.linspace(-numpy.pi, numpy.pi, 1000)
ax_phase.fill_between(phase_range, y_max, where=phase_range > 0, facecolor= utils.dna_rna_color_dict['RNA'], alpha=0.5, zorder=1)
ax_phase.fill_between(phase_range,y_max, where=phase_range < 0, facecolor= utils.dna_rna_color_dict['DNA'], alpha=0.5, zorder=1)

ax_phase.text(0.81, 0.8, 'RNA ' + r'$\rightarrow$' + ' DNA', fontsize=10, ha='center', va='center', transform=ax_phase.transAxes)
ax_phase.text(0.21, 0.8, 'DNA ' + r'$\rightarrow$' + ' RNA', fontsize=10, ha='center', va='center', transform=ax_phase.transAxes)

ax_phase.legend(loc='upper left', fontsize=6)


#ax_phase.fill_between(range(len(phase_range)), min(phase_range), max(phase_range), where=(phase_range < 0), alpha=0.5, color= utils.dna_rna_color_dict['RNA'])

#ax.fill_between(t, 1, where=s > 0, facecolor='green', alpha=.5)




fig.subplots_adjust(hspace=0.2, wspace=0.3)
fig_name = "%sfig3.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()




