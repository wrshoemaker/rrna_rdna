import config
import sys
import copy
import numpy
import utils
import random
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
from scipy.optimize import leastsq, curve_fit, minimize, brentq
from scipy.special import loggamma, gammaln, polygamma, digamma


import pickle

metadata_dict = utils.build_metadata_dict()
taxonomy_dict = utils.build_taxonomy_dict()
param_otu_mle_dict_path = config.data_directory + 'param_otu_mle_dict.pickle'

s_by_s, otu_labels, samples = utils.load_count_data()


#clr_s_by_s_dna, clr_s_by_s_rna, occupancy_idx, otu_labels_subset, n_reads_dna_occupancy, n_reads_rna_occupancy = utils.clr_transform(s_by_s, otu_labels, samples)

# get days
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

param_dict =  pickle.load(open(param_otu_mle_dict_path, 'rb'))

days = param_dict['data']['days']['RNA']
afd_rna = param_dict['data']['clr_afd']['RNA']
afd_dna = param_dict['data']['clr_afd']['DNA']
otu_labels = param_dict['otu_labels']

fig = plt.figure(figsize = (20, 20))
fig.subplots_adjust(bottom= 0.15)

idx_all = list(range(len(afd_dna)))
chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

asv_count = 0
for chunk_idx, chunk in enumerate(chunk_all):

    for c_idx, c in enumerate(chunk):

        ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

        afd_rna_c = afd_rna[c]
        afd_dna_c = afd_dna[c]

        rescaled_rna_c = (afd_rna_c - numpy.mean(afd_rna_c)) / numpy.std(afd_rna_c)
        rescaled_dna_c = (afd_dna_c - numpy.mean(afd_dna_c)) / numpy.std(afd_dna_c)

        #cv_rna_c = numpy.std(afd_rna_c)/numpy.median(numpy.abs(afd_rna_c - numpy.median(afd_rna_c)))
        #cv_dna_c = numpy.std(afd_dna_c)/numpy.median(numpy.abs(afd_dna_c - numpy.median(afd_dna_c)))

        x_rna, S_rna = utils.empirical_survival(rescaled_rna_c)
        x_dna, S_dna = utils.empirical_survival(rescaled_dna_c)
       

        ax.plot(x_rna, S_rna, ls='-', lw=3, c=utils.dna_rna_color_dict['RNA'], zorder=1, label='rRNA')
        ax.plot(x_dna, S_dna, ls='-', lw=3, c=utils.dna_rna_color_dict['DNA'], zorder=1, label='rDNA')

        #ax.plot(days_range, model_prediction, ls='-', lw=3, c=utils.dna_rna_color_dict[data_type], zorder=1, label=r'$\left\langle \mathrm{ln} \, x_{i}(t) \right\rangle$')

        #ax.scatter(days_c, afd_c, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type], zorder=2)
        #ax.axhline(y=0, ls=':', lw=2, zorder=0, c='k')#')
        #ax.set_xlabel("Time (days)", fontsize=10)
        #ax.set_ylabel("CLR-transformed abundance, " + utils.rescaled_label_clr_dict[data_type], fontsize=10)
        #ax.set_title('ASV %d (%s)' % (asv_count+1, taxonomy_dict[param_dict['otu_labels'][asv_count]]['family']), fontsize=11)
        
       
        #if (chunk_idx == 0) and (c_idx == 0):
        #    ax.legend(loc='upper right', fontsize=6)

        #asv_count += 1



fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%safd_per_asv.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
