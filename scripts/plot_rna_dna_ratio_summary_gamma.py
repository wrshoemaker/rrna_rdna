
import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import pickle
import sine_parameter_utils
import utils
import warnings
import config


from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import loggamma, gamma
from scipy.signal import fftconvolve
from scipy import stats, signal
from scipy.special import loggamma, gammaln, polygamma, digamma

import plot_autocorrelation_otu

# RNA:DNA distribution

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
taxonomy_dict = utils.build_taxonomy_dict()
otu_labels = param_dict['otu_labels']
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


def fit_loggamma(data):
    c, loc, scale = stats.loggamma.fit(data)
    return stats.loggamma(c, loc=loc, scale=scale)


def predict_log_ratio(rna_data, dna_data, n_points=4000):
    lg_rna = fit_loggamma(rna_data)
    lg_dna = fit_loggamma(dna_data)

    lo = min(lg_rna.ppf(1e-6), lg_dna.ppf(1e-6))
    hi = max(lg_rna.ppf(1-1e-6), lg_dna.ppf(1-1e-6))
    grid = numpy.linspace(lo, hi, n_points)
    dx   = grid[1] - grid[0]

    pdf_rna = lg_rna.pdf(grid)
    pdf_dna = lg_dna.pdf(grid)

    # D = log(RNA) - log(DNA) =>  convolve with flipped DNA pdf
    pdf_diff = fftconvolve(pdf_rna, pdf_dna[::-1], mode='full') * dx
    d_grid   = numpy.linspace(lo - hi, hi - lo, len(pdf_diff))

    return d_grid, pdf_diff, lg_rna, lg_dna



afd_dna_all = []
afd_rna_all = []
for otu_idx in range(len(otu_labels)):

    afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_idx])
    afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_idx])
    afd_rna_dna = afd_rna - afd_dna

    afd_dna_all.append(afd_dna)
    afd_rna_all.append(afd_rna)


afd_dna_all = numpy.concatenate(afd_dna_all)
afd_rna_all = numpy.concatenate(afd_rna_all)
afd_rna_dna_all = afd_rna_all - afd_dna_all
rescaled_afd_rna_dna_all = (afd_rna_dna_all - numpy.mean(afd_rna_dna_all)) / numpy.std(afd_rna_dna_all)
d_grid, pdf_diff, lg_rna, lg_dna = predict_log_ratio(afd_rna_all, afd_dna_all)
# if Z = (D - mu) / std ==> f_Z(z) = std * f_D(std*z + mu)
mu  = numpy.mean(afd_rna_dna_all)
std = numpy.std(afd_rna_dna_all)

d_grid_z   = (d_grid - mu) / std
pdf_diff_z  = pdf_diff * std  


#### time vs. mean
otu_idx = 0
days_dna_c = numpy.asarray(param_dict['data']['days']['DNA'][otu_idx])
afd_dna_c = numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_idx])
days_rna_c = numpy.asarray(param_dict['data']['days']['RNA'][otu_idx])
afd_rna_c = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_idx])
otu_label = param_dict['otu_labels'][otu_idx]


method = 'mle'
amp_dna_c = param_dict['amp_%s' % method]['DNA'][otu_idx]
amp_rna_c = param_dict['amp_%s' % method]['RNA'][otu_idx]
freq_dna_c = param_dict['freq_%s' % method]['DNA'][otu_idx]
freq_rna_c = param_dict['freq_%s' % method]['RNA'][otu_idx]
phase_dna_c = param_dict['phase_%s' % method]['DNA'][otu_idx]
phase_rna_c = param_dict['phase_%s' % method]['RNA'][otu_idx]
param_mean_dna_c = param_dict['param_mean_%s' % method]['DNA'][otu_idx]
param_mean_rna_c = param_dict['param_mean_%s' % method]['RNA'][otu_idx]
sigma_dna_c = param_dict['sigma_corrected']['DNA'][otu_idx]
sigma_rna_c = param_dict['sigma_corrected']['DNA'][otu_idx]
beta_dna_c = (2 - sigma_dna_c) / sigma_dna_c
beta_rna_c = (2 - sigma_rna_c) / sigma_rna_c

M_dna = (digamma(beta_dna_c) - numpy.log(beta_dna_c) + numpy.log(1 - sigma_dna_c / 2) + numpy.log(param_mean_dna_c))
M_rna = (digamma(beta_rna_c) - numpy.log(beta_rna_c) + numpy.log(1 - sigma_rna_c / 2) + numpy.log(param_mean_rna_c))
delta_M = M_rna - M_dna
timescale_dna_c = 2*numpy.pi/freq_dna_c
timescale_rna_c = 2*numpy.pi/freq_rna_c
diff_afd_c = afd_rna_c - afd_dna_c
days_range = numpy.linspace(min(days_dna_c), max(days_dna_c), 1000)
K_dna = amp_dna_c * numpy.sin(2 * numpy.pi * days_range / timescale_dna_c + phase_dna_c)
K_rna = amp_rna_c * numpy.sin(2 * numpy.pi * days_range / timescale_rna_c + phase_rna_c)
expected_rna_dna = delta_M + K_rna - K_dna


#####autocorrr
autocorr_dict = pickle.load(open(plot_autocorrelation_otu.autocorrelation_dict_path, "rb"))
delta_t_c = autocorr_dict['otu'][otu_labels[otu_idx]]['ratio']['delta_t']
autocorr_obs_c = autocorr_dict['otu'][otu_labels[otu_idx]]['ratio']['autocorr_obs']
autocorr_pred_c = autocorr_dict['otu'][otu_labels[otu_idx]]['ratio']['autocorr_pred']


fig = plt.figure(figsize=(8.5, 8))
gs = GridSpec(2, 2, figure=fig)

ax_dist = fig.add_subplot(gs[0, 1])
ax_mean = fig.add_subplot(gs[1, 0])
ax_auto = fig.add_subplot(gs[1, 1])

# plot dist
counts, bins = numpy.histogram(rescaled_afd_rna_dna_all, bins=30)
midpoints = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]
expected_counts = pdf_diff_z * bin_width * len(rescaled_afd_rna_dna_all)

ax_dist.scatter(midpoints, counts, s=80, facecolors='none', edgecolors='k', alpha=1, linewidths=2, label='Observed')
ax_dist.plot(d_grid_z, expected_counts, color='k', lw=4, ls=':', label='Predicted')

ax_dist.set_yscale('log', base=10)
ax_dist.set_ylim([min(counts)/1.3, max(counts)*1.3])
#ax.set_xlabel('standardised log(RNA) − log(DNA)')
#ax.set_ylabel('Density')

ax_dist.set_xlabel('Rescaled rRNA:rDNA, ' + r'$\phi$', fontsize=14)
ax_dist.set_ylabel('Probability density', fontsize=14)
#  framealpha=1,
ax_dist.legend(loc='lower right', fontsize=9)
ax_dist.tick_params(axis='x', labelsize=7)
ax_dist.tick_params(axis='y', labelsize=7)
ax_dist.set_title('All ASVs', fontsize=14)


# expected value
ax_mean.scatter(days_rna_c, diff_afd_c, s=8, alpha=1, c=utils.dna_rna_color_dict['ratio'], zorder=1, label='Observed')
ax_mean.plot(days_range, expected_rna_dna, ls='-', lw=3, c=utils.dna_rna_color_dict['ratio'], zorder=2,  label='Predicted')
ax_mean.set_xlabel("Time (days), " + r'$t$' , fontsize=14)
ax_mean.set_ylabel("rRNA:rDNA, " + r'$\phi(t)$', fontsize=14)
ax_mean.set_title('ASV %d (%s)' % (1, taxonomy_dict[otu_labels[otu_idx]]['family']), fontsize=14)

#minor_days, major_days, major_labels
ax_mean.set_xlim([0, max(days_dna_c)])
ax_mean.set_xticks(minor_days, minor=True)
ax_mean.set_xticks(major_days, minor=False)
ax_mean.set_xticklabels(major_labels, minor=False, fontsize=7)
#max_ = numpy.absolute(max(residuals))
ax_mean.axhline(y=0, lw=2, ls=':', c='k')
#ax.set_ylim([-1*max_, max_])
ax_mean.legend(loc='lower right', fontsize=9)
ax_mean.tick_params(axis='y', labelsize=7)


# plot autocorrelation
ax_auto.scatter(delta_t_c, autocorr_obs_c, s=7, alpha=1, zorder=1, c=utils.dna_rna_color_dict['ratio'], label='Observed')
ax_auto.plot(delta_t_c, autocorr_pred_c, ls='-', lw=3, zorder=2, c=utils.dna_rna_color_dict['ratio'], label='Predicted')
ax_auto.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 14)
ax_auto.set_ylabel(utils.sample_label_dict['ratio'] + " autocorrelation", fontsize = 14)
#ax.set_title(otu_labels[c], fontsize=11)
ax_auto.set_title('ASV %d (%s)' % (1, taxonomy_dict[otu_labels[otu_idx]]['family']), fontsize=14)
ax_auto.legend(loc='lower right', fontsize=9)
ax_auto.tick_params(axis='x', labelsize=7)
ax_auto.tick_params(axis='y', labelsize=7)



fig.subplots_adjust(hspace=0.3, wspace=0.3)
fig_name = "%srna_dna_ratio_summary.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


