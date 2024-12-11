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
from scipy.stats import gamma, loggamma


import plot_predict_change_dna


numpy.random.seed(123456789)

n_iter = int(1e4)
method = 'mle'

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
null_predict_change_dict = plot_predict_change_dna.load_null_predict_change_dict_path()

days = numpy.asarray(param_dict['data']['days']['RNA'][0])

focal_otu = 'Otu000001'
#focal_otu_formatted = 'OTU 1'
focal_otu_formatted = 'OTU 1 ('+ r'$\mathit{Anabaena}$' + ' sp.)'

focal_otu_idx = 0


# parameters for OTU1
amp_dna = param_dict['amp_%s' % method]['DNA'][focal_otu_idx]
amp_rna = param_dict['amp_%s' % method]['RNA'][focal_otu_idx]

freq_dna = param_dict['freq_%s' % method]['DNA'][focal_otu_idx]
freq_rna = param_dict['freq_%s' % method]['RNA'][focal_otu_idx]

phase_dna = param_dict['phase_%s' % method]['DNA'][focal_otu_idx]
phase_rna = param_dict['phase_%s' % method]['RNA'][focal_otu_idx]

param_mean_dna = param_dict['param_mean_%s' % method]['DNA'][focal_otu_idx]
param_mean_rna = param_dict['param_mean_%s' % method]['RNA'][focal_otu_idx]

sigma_dna =  param_dict['sigma']['DNA'][focal_otu_idx]
sigma_rna =  param_dict['sigma']['RNA'][focal_otu_idx]

mean_dna = param_mean_dna * numpy.exp(amp_dna * numpy.sin(freq_dna*days + phase_dna))
mean_rna = param_mean_rna * numpy.exp(amp_rna * numpy.sin(freq_rna*days + phase_rna))

k_dna = mean_dna / (1 - (sigma_dna/2))
k_rna = mean_rna / (1 - (sigma_rna/2))


def sine_slope_null():

    slope_all = []
    for n in range(n_iter):

        exp_clr_sim_dna = gamma.rvs(numpy.divide(2,sigma_dna)-1, scale=sigma_dna*k_dna/2, size=len(days))
        exp_clr_sim_rna = gamma.rvs(numpy.divide(2,sigma_rna)-1, scale=sigma_rna*k_rna/2, size=len(days))

        clr_sim_dna = numpy.log(exp_clr_sim_dna)
        clr_sim_rna = numpy.log(exp_clr_sim_rna)

        diff_rna_dna = clr_sim_rna - clr_sim_dna

        slope, intercept, r_value, p_value, std_err = stats.linregress(diff_rna_dna[:-1], clr_sim_dna[1:])

        slope_all.append(slope)


    slope_all = numpy.asarray(slope_all)

    return slope_all



fig = plt.figure(figsize = (8.5, 4)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=1, ncols=2)

ax_scatter = fig.add_subplot(gs[0, 0])
ax_sine_null = fig.add_subplot(gs[0, 1])

clr_s_by_s_rescaled_ratio = numpy.asarray(null_predict_change_dict[focal_otu]['clr_s_by_s_rescaled_ratio'])
diff_clr_s_by_s_rescaled_dna = numpy.asarray(null_predict_change_dict[focal_otu]['diff_clr_s_by_s_rescaled_dna'])

ax_scatter.scatter(clr_s_by_s_rescaled_ratio, diff_clr_s_by_s_rescaled_dna, s=8, alpha=1, c='k', zorder=2)
ax_scatter.set_xlabel("Difference between RNA and DNA at time " + r'$t$', fontsize=10)
ax_scatter.set_ylabel("DNA at time " + r'$t+\delta t$', fontsize=10)
ax_scatter.set_title( focal_otu_formatted + '\nNull: Time label permutation', fontsize=11)

slope, intercept, r_value, p_value, std_err = stats.linregress(clr_s_by_s_rescaled_ratio, diff_clr_s_by_s_rescaled_dna)
            
x_range_ =  numpy.linspace(min(clr_s_by_s_rescaled_ratio), max(clr_s_by_s_rescaled_ratio), 10000)
y_fit_range = slope*x_range_ + intercept
ax_scatter.plot(x_range_, y_fit_range, ls='--', lw=2.5, c='k')

ax_scatter.text(0.26, 0.78, utils.get_p_value_latex_label_dict(0.0001), fontsize=12, ha='center', va='center', transform=ax_scatter.transAxes)
ax_scatter.text(0.26, 0.87, 'Slope = ' + str(round(slope, 3)), fontsize=12, ha='center', va='center', transform=ax_scatter.transAxes)


sine_slope_null = sine_slope_null()

ax_sine_null.hist(sine_slope_null, bins=50, lw=2, color=utils.dna_rna_color_dict['ratio'], histtype='step', density=True, alpha=0.8, zorder=1, label='Null')
ax_sine_null.axvline(x=slope, lw=3, ls='--', c='k', zorder=2, label='Observed')
#ax.set_xlim([-0.55,0.55])
ax_sine_null.set_title( focal_otu_formatted + '\nNull: Gamma with oscillating ' + r'$K_{i}(t)$', fontsize=11)
ax_sine_null.legend(loc='upper left')
ax_sine_null.set_xlabel("Slope between RNA - DNA at time " + r'$t$' + '\nand DNA at time ' r'$t+\delta t$', fontsize=10)
ax_sine_null.set_ylabel("Probability density", fontsize=10)





fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig_name = "%sfig4.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



