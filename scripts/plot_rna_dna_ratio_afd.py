import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from statsmodels.stats.multitest import fdrcorrection

from scipy.stats import loggamma, gamma
from scipy.signal import fftconvolve


from scipy import stats, signal
# numdifftools also installed
import pickle

import sine_parameter_utils



param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
taxonomy_dict = utils.build_taxonomy_dict()
otu_labels = param_dict['otu_labels']

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

    print(numpy.std(afd_dna) / numpy.std(afd_rna))

    #rescaled_afd_dna = (afd_dna - numpy.mean(afd_dna)) / numpy.std(afd_dna)
    #rescaled_afd_rna = (afd_rna - numpy.mean(afd_rna)) / numpy.std(afd_rna)
    #afd_rna_dna_rescaled = (afd_rna_dna - numpy.mean(afd_rna_dna)) / numpy.std(afd_rna_dna)

    #d_grid, pdf_diff, lg_rna, lg_dna = predict_log_ratio(rescaled_afd_rna, rescaled_afd_dna)

    # standardize the predicted ratio, not the inputs
    #mean_D = np.average(d_grid, weights=pdf_diff)
    #var_D  = np.average((d_grid - mean_D)**2, weights=pdf_diff)
    #std_D  = np.sqrt(var_D)

    #d_grid_z  = (d_grid - mean_D) / std_D

    #mask = np.isfinite(rna) & (rna > 0) & np.isfinite(dna) & (dna > 0)
    #empirical = np.log(rna[mask]) - np.log(dna[mask])

    #fig, ax = plt.subplots(figsize=(4.5, 4))
    #ax.hist(afd_rna_dna, bins=50, density=True, color='k', alpha=0.4)

    #ax.plot(d_grid, pdf_diff, color='crimson', lw=2, label='Predicted (loggamma convolution)')


    #print(pdf_diff)


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


fig, ax = plt.subplots(figsize=(4.5, 4))
counts, bins = numpy.histogram(rescaled_afd_rna_dna_all, bins=30)
midpoints = (bins[:-1] + bins[1:]) / 2
bin_width = bins[1] - bins[0]

expected_counts = pdf_diff_z * bin_width * len(rescaled_afd_rna_dna_all)

ax.scatter(midpoints, counts, s=80, facecolors='none', edgecolors='k', alpha=1, linewidths=2, label='Observed')
ax.plot(d_grid_z, expected_counts, color='k', lw=4, ls=':', label='Predicted')

ax.set_yscale('log', base=10)
ax.set_ylim([min(counts)/1.3, max(counts)*1.3])
#ax.set_xlabel('standardised log(RNA) − log(DNA)')
#ax.set_ylabel('Density')

ax.set_xlabel('Rescaled rRNA:rDNA, ' + r'$\phi$', fontsize=12)
ax.set_ylabel('Probability density', fontsize=12)
ax.legend(loc='upper left')

fig.subplots_adjust(hspace=0.3, wspace=0.35)
fig_name = "%srna_dna_ratio_afd.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
