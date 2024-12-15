import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from statsmodels.stats.multitest import fdrcorrection

from scipy import stats, signal
# numdifftools also installed
import pickle

import sine_parameter_utils

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))


def build_ll_dict():

    #ll_dict = {}

    otu_labels = param_dict['otu_labels']

    ll_dna = []
    ll_rna = []
    
    pvalue_dna = []
    pvalue_rna = []

    for otu_label_i_idx, otu_label_i in enumerate(otu_labels):

        #ll_dict[otu_label_i] = {}

        afd_dna_i = numpy.exp(numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_label_i_idx]))
        afd_rna_i = numpy.exp(numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_label_i_idx]))

        days_dna_i = numpy.asarray(param_dict['data']['days']['DNA'][otu_label_i_idx])
        days_rna_i = numpy.asarray(param_dict['data']['days']['RNA'][otu_label_i_idx])
        
        beta_dna_i = param_dict['beta']['DNA'][otu_label_i_idx]
        beta_rna_i = param_dict['beta']['RNA'][otu_label_i_idx]

        mean_gamma_dna_i = numpy.mean(numpy.exp(afd_dna_i))
        mean_gamma_rna_i = numpy.mean(numpy.exp(afd_rna_i))

        ll_gamma_dna_i = sine_parameter_utils.ll_gamma(afd_dna_i, mean_gamma_dna_i, beta_dna_i)
        ll_gamma_rna_i = sine_parameter_utils.ll_gamma(afd_rna_i, mean_gamma_rna_i, beta_rna_i)

        params_dna_dict = {'amp': param_dict['amp_mle']['DNA'][otu_label_i_idx], 'freq': param_dict['freq_mle']['DNA'][otu_label_i_idx], 'phase': param_dict['phase_mle']['DNA'][otu_label_i_idx], 'param_mean': param_dict['param_mean_mle']['DNA'][otu_label_i_idx]}
        params_rna_dict = {'amp': param_dict['amp_mle']['RNA'][otu_label_i_idx], 'freq': param_dict['freq_mle']['RNA'][otu_label_i_idx], 'phase': param_dict['phase_mle']['RNA'][otu_label_i_idx], 'param_mean': param_dict['param_mean_mle']['RNA'][otu_label_i_idx]}

        ll_sine_gamma_dna_i = sine_parameter_utils.ll_sine_gamma(params_dna_dict, days_dna_i, afd_dna_i, beta_dna_i)
        ll_sine_gamma_rna_i = sine_parameter_utils.ll_sine_gamma(params_rna_dict, days_rna_i, afd_rna_i, beta_rna_i)

        ll_gamma_dna_i = -1*ll_gamma_dna_i
        ll_gamma_rna_i = -1*ll_gamma_rna_i
        ll_sine_gamma_dna_i = -1*ll_sine_gamma_dna_i
        ll_sine_gamma_rna_i = -1*ll_sine_gamma_rna_i

        #param_n_diff = 
        # sine gamma model has 5 parameters, gamma has 2
        # df = 5 - 2 = 3

        lr_dna_i =  -2*(ll_gamma_dna_i - ll_sine_gamma_dna_i)
        lr_rna_i =  -2*(ll_gamma_rna_i - ll_sine_gamma_rna_i)

        pvalue_dna_i = stats.distributions.chi2.sf(lr_dna_i, 3)
        pvalue_rna_i = stats.distributions.chi2.sf(lr_rna_i, 3)

        ll_dna.append(lr_dna_i)
        ll_rna.append(lr_rna_i)
        
        pvalue_dna.append(pvalue_dna_i)
        pvalue_rna.append(pvalue_rna_i)


    ll_dna = numpy.asarray(ll_dna)
    ll_rna = numpy.asarray(ll_rna)
    
    pvalue_dna = numpy.asarray(pvalue_dna)
    pvalue_rna = numpy.asarray(pvalue_rna)

    # set p values of zero to lowers p value
    pvalue_dna[pvalue_dna == float(0)] = min(pvalue_dna[pvalue_dna > float(0)])
    pvalue_rna[pvalue_rna == float(0)] = min(pvalue_rna[pvalue_rna > float(0)])


    return numpy.asarray(otu_labels), ll_dna, ll_rna, pvalue_dna, pvalue_rna




otu_labels, ll_dna, ll_rna, pvalue_dna, pvalue_rna = build_ll_dict()

# BH correction 

pvalue_dna = fdrcorrection(pvalue_dna, alpha=0.05, method='indep', is_sorted=False)[1]
pvalue_rna = fdrcorrection(pvalue_rna, alpha=0.05, method='indep', is_sorted=False)[1]

pvalue_dna_log10 = -1*numpy.log10(pvalue_dna)
pvalue_rna_log10 = -1*numpy.log10(pvalue_rna)




y_axis_idx = numpy.asarray(range(len(otu_labels)))


fig, ax = plt.subplots(figsize=(4,4))


ax.set_yticks(y_axis_idx)
ax.set_yticklabels(otu_labels, fontsize=8)


ax.scatter(pvalue_dna_log10, y_axis_idx, alpha=0.7, s=30, color=utils.dna_rna_color_dict['DNA'], label='DNA')
ax.scatter(pvalue_rna_log10, y_axis_idx, alpha=0.7, s=30, color=utils.dna_rna_color_dict['RNA'], label='RNA')


ax.set_xlabel(r'$- \mathrm{log}_{10}P$', fontsize=12)
ax.axvline(x=-1*numpy.log10(0.05), lw=2.5, ls=':', label=r'$P = 0.05$', color='k', zorder=1)
ax.legend(loc='lower left', fontsize=6)


fig.subplots_adjust(hspace=0.35, wspace=0.25)
fig_name = "%sll.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



