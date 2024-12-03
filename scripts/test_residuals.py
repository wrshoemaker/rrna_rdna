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



focal_otu = 'Otu000001'
focal_otu_idx = param_dict['otu_labels'].index('Otu000001')


days_focal = numpy.asarray(param_dict['data']['days']['DNA'][focal_otu_idx])
afd_dna_focal = numpy.asarray(param_dict['data']['clr_afd']['DNA'][focal_otu_idx])
afd_rna_focal = numpy.asarray(param_dict['data']['clr_afd']['RNA'][focal_otu_idx])

days_range = numpy.linspace(min(days_focal), max(days_focal), 1000)
#model_prediction = amp_focal*numpy.sin(freq_focal*days_range+phase_focal) + numpy.log(param_mean_focal)
model_prediction_dna = param_dict['amp_%s' % method]['DNA'][focal_otu_idx]*numpy.sin(param_dict['freq_%s' % method]['DNA'][focal_otu_idx]*days_focal+param_dict['phase_%s' % method]['DNA'][focal_otu_idx]) + numpy.log(param_dict['param_mean_%s' % method]['DNA'][focal_otu_idx]) + numpy.log(1 - (2/(param_dict['beta']['DNA'][focal_otu_idx]+1))/2)
model_prediction_rna = param_dict['amp_%s' % method]['RNA'][focal_otu_idx]*numpy.sin(param_dict['freq_%s' % method]['RNA'][focal_otu_idx]*days_focal+param_dict['phase_%s' % method]['RNA'][focal_otu_idx]) + numpy.log(param_dict['param_mean_%s' % method]['RNA'][focal_otu_idx]) + numpy.log(1 - (2/(param_dict['beta']['RNA'][focal_otu_idx]+1))/2)

resid_dna = afd_dna_focal - model_prediction_dna
resid_ratio = (afd_rna_focal - afd_dna_focal) - (model_prediction_rna - model_prediction_dna)


fig = plt.figure(figsize = (4.5, 8))
fig.subplots_adjust(bottom= 0.15)

ax_resid = plt.subplot2grid((2, 1), (0, 0))
ax_regress = plt.subplot2grid((2, 1), (1, 0))

#fig, ax = plt.subplots(figsize=(6,4))
#ax.scatter(resid_ratio[:-1], resid_dna[1:])

ax_resid.scatter(days_focal, resid_dna, c=utils.dna_rna_color_dict['DNA'], label='DNA', s=14)
ax_resid.scatter(days_focal, resid_ratio, c='k', label='RNA:DNA', s=14)
ax_resid.legend(loc='upper left')

ax_resid.set_xlabel('Time (days)')
ax_resid.set_ylabel('Residuals')


ax_regress.scatter(resid_ratio[:-1], resid_dna[1:])

ax_regress.set_xlabel('RNA:DNA residuals at time t')
ax_regress.set_ylabel('DNA residuals at time t+delta t')

slope, intercept, r_value, p_value, std_err = stats.linregress(resid_ratio[:-1], resid_dna[1:])

print(slope, p_value)

x_range_ =  numpy.linspace(min(resid_ratio), max(resid_ratio), 10000)
y_fit_range = slope*x_range_ + intercept
ax_regress.plot(x_range_, y_fit_range, ls='--', lw=2.5, c='k')


fig.subplots_adjust(hspace=0.45, wspace=0.35)
fig_name = "%stest_resid.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

