import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
import sine_parameter_utils


method = 'leastsq'


s_by_s, otu_labels, samples = utils.load_count_data()
#rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples, min_occupancy=1)


metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


param_dict = sine_parameter_utils.load_param_otu_dict(log10_status=False, clr_status=True)


# calculate_sine_wave(t, amp, freq, phase, param_mean)

otu='Otu000001'
otu_idx = param_dict['otu_labels'].index(otu)

#sine_parameter_utils.calculate_sine_wave(days, )

amp_dna = param_dict['amp_%s' % (method)]['DNA'][otu_idx]
freq_dna = param_dict['freq_%s' % (method)]['DNA'][otu_idx]
phase_dna = param_dict['phase_%s' % (method)]['DNA'][otu_idx]
param_mean_dna = param_dict['param_mean_%s' % (method)]['DNA'][otu_idx]

amp_rna = param_dict['amp_%s' % (method)]['RNA'][otu_idx]
freq_rna = param_dict['freq_%s' % (method)]['RNA'][otu_idx]
phase_rna = param_dict['phase_%s' % (method)]['RNA'][otu_idx]
param_mean_rna = param_dict['param_mean_%s' % (method)]['RNA'][otu_idx]

afd_dna = sine_parameter_utils.calculate_sine_wave(days, amp_dna, freq_dna, phase_dna, param_mean_dna)
afd_rna = sine_parameter_utils.calculate_sine_wave(days, amp_rna, freq_rna, phase_rna, param_mean_rna)


#afd_dna = afd_dna - param_mean_dna
#afd_rna = afd_rna - param_mean_rna

rna_dna_ratio = (afd_rna - afd_dna)[1:]
delta_afd_dna = afd_dna[1:] - afd_dna[:-1]


fig, ax = plt.subplots(figsize=(4,4))


ax.scatter(rna_dna_ratio, delta_afd_dna, s=10)

fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%srna_dna_ratio_diff_sine.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


