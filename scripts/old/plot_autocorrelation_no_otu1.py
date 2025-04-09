import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
# numdifftools also installed
import pickle

import plot_sine_parameters


numpy.seterr(divide='ignore', invalid='ignore')
min_n_obs = 10


#def autocorrelation(tau, delta_t):

#label_dict = {'DNA':  r'$R_{\tilde{X}_{i}}(\Delta t)$'}

# 

def plot_autocorrelation_otu(data_type):

    metadata_dict = utils.build_metadata_dict()

    s_by_s, otu_labels, samples = utils.load_count_data()
    s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

    otu1_idx = otu_labels_subset == 'Otu000001'
    s_by_s_dna_no_otu1 = s_by_s_dna[~otu1_idx,:]
    s_by_s_rna_no_otu1 = s_by_s_rna[~otu1_idx,:]
    otu_labels_subset_no_otu1 = otu_labels_subset[~otu1_idx]

    # returns rescaled relative abundance
    rel_s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
    rel_s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)
    rel_s_by_s_rescaled_ratio = rel_s_by_s_rescaled_rna/rel_s_by_s_rescaled_dna

    rel_s_by_s_rescaled_dna_log = numpy.log10(rel_s_by_s_rescaled_dna)
    rel_s_by_s_rescaled_rna_log = numpy.log10(rel_s_by_s_rescaled_rna)
    rel_s_by_s_rescaled_ratio_log = numpy.log10(rel_s_by_s_rescaled_ratio)

    # no otu1
    rel_s_by_s_rescaled_dna_no_otu1 = utils.rescale_s_by_s(s_by_s_dna_no_otu1)
    rel_s_by_s_rescaled_rna_no_otu1 = utils.rescale_s_by_s(s_by_s_rna_no_otu1)
    rel_s_by_s_rescaled_ratio_no_otu1 = rel_s_by_s_rescaled_rna_no_otu1/rel_s_by_s_rescaled_dna_no_otu1

    rel_s_by_s_rescaled_dna_log_no_otu1 = numpy.log10(rel_s_by_s_rescaled_dna_no_otu1)
    rel_s_by_s_rescaled_rna_log_no_otu1 = numpy.log10(rel_s_by_s_rescaled_rna_no_otu1)
    rel_s_by_s_rescaled_ratio_log_no_otu1 = numpy.log10(rel_s_by_s_rescaled_ratio_no_otu1)


    # get days
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    #idx_all = list(range(len(otu_labels_subset)))
    chunk_all = [otu_labels_subset_no_otu1[x:x+5] for x in range(0, len(otu_labels_subset_no_otu1), 5)]

    n_timepoints = rel_s_by_s_rescaled_rna.shape[1]
    time_increments = list(range(1, n_timepoints-min_n_obs+1))
    delta_t = numpy.asarray([days[i] - days[0] for i in time_increments])

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            idx_c = numpy.where(otu_labels_subset==c)[0][0]
            idx_c_no_otu1 = numpy.where(otu_labels_subset_no_otu1==c)[0][0]

            if data_type == 'DNA':
                afd_c = rel_s_by_s_rescaled_dna_log[idx_c,:]
                afd_c_no_otu1 = rel_s_by_s_rescaled_dna_log_no_otu1[idx_c_no_otu1,:]

            elif data_type == 'RNA':
                afd_c = rel_s_by_s_rescaled_rna_log[idx_c,:]
                afd_c_no_otu1 = rel_s_by_s_rescaled_rna_log_no_otu1[idx_c_no_otu1,:]

            elif data_type == 'ratio':
                afd_c = rel_s_by_s_rescaled_ratio_log[idx_c,:]
                afd_c_no_otu1 = rel_s_by_s_rescaled_ratio_log_no_otu1[idx_c_no_otu1,:]

            else:
                sys.stderr.write("Argument not recognized!\n")
                sys.exit()


            # rescale using sine fits
                        
            autocorr_obs_c = [numpy.corrcoef(afd_c[i:], afd_c[:-i])[0,1] for i in time_increments]
            autocorr_obs_c_no_otu1 = [numpy.corrcoef(afd_c_no_otu1[i:], afd_c_no_otu1[:-i])[0,1] for i in time_increments]
            #autocorr_pred_c = 0.5*numpy.cos((2*numpy.pi*delta_t)/freq_leastsq_all[c])

            #ax.scatter(delta_t, autocorr_obs_c, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type], label='Observed')
            ax.plot(delta_t, autocorr_obs_c, ls='-', lw=2, c=utils.dna_rna_color_dict[data_type], label='All', zorder=1)
            ax.plot(delta_t, autocorr_obs_c_no_otu1, ls=':', lw=2, c=utils.dna_rna_color_dict[data_type], label='No OTU1', zorder=2)

            ax.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 10)
            ax.set_ylabel("Autocorrelation, " + utils.sample_label_dict[data_type], fontsize = 10)

            ax.set_title(c, fontsize=11)

            if (chunk_idx==0) and (c_idx==0):
                ax.legend(loc='upper left', fontsize=8)
    

    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%sautocorrelation_no_otu1_%s.png" % (config.analysis_directory, data_type)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




plot_autocorrelation_otu('DNA')
plot_autocorrelation_otu('RNA')
plot_autocorrelation_otu('ratio')