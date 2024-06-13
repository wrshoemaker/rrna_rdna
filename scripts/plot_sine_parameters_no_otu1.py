

import config
import sys
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
from scipy.optimize import leastsq, curve_fit, minimize
from lmfit import Minimizer, create_params, fit_report
# numdifftools also installed
import pickle
import plot_sine_parameters



param_dict = plot_sine_parameters.load_param_dict(True, None)
param_no_otu1_dict = plot_sine_parameters.load_param_dict(True, 'Otu000001')


otu_labels = numpy.asarray(param_dict['otu']['otu_labels'])
otu_labels_no_otu1 = numpy.asarray(param_no_otu1_dict['otu']['otu_labels'])
otu_labels_intersect = numpy.intersect1d(otu_labels, otu_labels_no_otu1)

to_plot_all_idx = numpy.asarray([numpy.where(otu_labels==i)[0][0] for i in otu_labels_intersect])
to_plot_no_otu1_idx = numpy.asarray([numpy.where(otu_labels_no_otu1==i)[0][0] for i in otu_labels_intersect])


fig = plt.figure(figsize = (12, 16))
fig.subplots_adjust(bottom= 0.15)


for data_type_idx, data_type in enumerate(['DNA', 'RNA', 'ratio']):

    amp_all = numpy.asarray(param_dict['otu']['amp_leastsq'][data_type])[to_plot_all_idx]
    amp_no_otu1 = numpy.asarray(param_no_otu1_dict['otu']['amp_leastsq'][data_type])[to_plot_no_otu1_idx]
    amp_merged = numpy.concatenate([amp_all, amp_no_otu1])
    amp_min_max = [min(amp_merged), max(amp_merged)]

    freq_all = numpy.asarray(param_dict['otu']['freq_leastsq'][data_type])[to_plot_all_idx]
    freq_no_otu1 = numpy.asarray(param_no_otu1_dict['otu']['freq_leastsq'][data_type])[to_plot_no_otu1_idx]
    freq_merged = numpy.concatenate([freq_all, freq_no_otu1])
    freq_min_max = [min(freq_merged), max(freq_merged)]

    phase_all = numpy.asarray(param_dict['otu']['phase_leastsq'][data_type])[to_plot_all_idx]
    phase_no_otu1 = numpy.asarray(param_no_otu1_dict['otu']['phase_leastsq'][data_type])[to_plot_no_otu1_idx]
    phase_merged = numpy.concatenate([phase_all, phase_no_otu1])
    phase_min_max = [min(phase_merged), max(phase_merged)]

    #scaled_time_all = freq_all+freq_merged
    scaled_time_merged = numpy.concatenate([freq_all+phase_all, freq_no_otu1+phase_no_otu1])
    scaled_time_min_max = [min(scaled_time_merged), max(scaled_time_merged)]
    

    ax_amp = plt.subplot2grid((4, 3), (0, data_type_idx))
    ax_freq = plt.subplot2grid((4,3), (1, data_type_idx))
    ax_phase = plt.subplot2grid((4, 3), (2, data_type_idx))
    ax_scaled_time = plt.subplot2grid((4, 3), (3, data_type_idx))

    ax_amp.set_title(data_type, fontsize=14)

    ax_amp.scatter(amp_all, amp_no_otu1, s=15, alpha=0.9, c=utils.dna_rna_color_dict[data_type], zorder=2)
    ax_amp.plot(amp_min_max, amp_min_max, ls=':', lw=1.5, c='k', zorder=1, label='1:1')
    ax_amp.set_xlim(amp_min_max)
    ax_amp.set_ylim(amp_min_max)
    ax_amp.set_xlabel('Amplitude, ' + r'$A_{i}$', fontsize=10)
    ax_amp.set_ylabel('Amplitude excluding dominant OTU, ' + r'$A_{i}$', fontsize=10)

    if (data_type_idx ==0):
        ax_amp.legend(loc='upper left', fontsize=11)


    ax_freq.scatter(freq_all, freq_no_otu1, s=15, alpha=0.9, c=utils.dna_rna_color_dict[data_type], zorder=2)
    ax_freq.plot(freq_min_max, freq_min_max, ls=':', lw=1.5, c='k', zorder=1)
    ax_freq.set_xlim(freq_min_max)
    ax_freq.set_ylim(freq_min_max)
    ax_freq.set_xlabel('Frequency, ' + r'$ \frac{2\pi}{\tau_{i}}$', fontsize=10)
    ax_freq.set_ylabel('Frequency excluding dominant OTU, ' + r'$ \frac{2\pi}{\tau_{i}}$', fontsize=10)
    #ax_freq.set_xscale('log', basex=10)
    #ax_freq.set_yscale('log', basey=10)


    ax_phase.scatter(phase_all, phase_no_otu1, s=15, alpha=0.9, c=utils.dna_rna_color_dict[data_type], zorder=2)
    ax_phase.plot(phase_min_max, phase_min_max, ls=':', lw=1.5, c='k', zorder=1, label='1:1')
    ax_phase.set_xlim(phase_min_max)
    ax_phase.set_ylim(phase_min_max)
    ax_phase.set_xlabel('Phase, ' + r'$\psi_{i}$', fontsize=10)
    ax_phase.set_ylabel('Phase exclding dominant OTU, ' + r'$\psi_{i}$', fontsize=10)


    ax_scaled_time.scatter(freq_all+phase_all, freq_no_otu1+phase_no_otu1, s=15, alpha=0.9, c=utils.dna_rna_color_dict[data_type], zorder=2)
    ax_scaled_time.plot(scaled_time_min_max, scaled_time_min_max, ls=':', lw=1.5, c='k', zorder=1, label='1:1')
    ax_scaled_time.set_xlim(scaled_time_min_max)
    ax_scaled_time.set_ylim(scaled_time_min_max)
    ax_scaled_time.set_xlabel('Scaled time, ' + r'$ \frac{2\pi}{\tau_{i}} + \psi_{i}$', fontsize=10)
    ax_scaled_time.set_ylabel('Scaled time excluding dominant OTU, ' + r'$ \frac{2\pi}{\tau_{i}} + \psi_{i}$', fontsize=10)



fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%ssine_parameter_no_otu1.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


#print(param_no_otu1_dict['otu']['amp_leastsq'].keys())




#['amp_leastsq', 'freq_leastsq', 'phase_leastsq']