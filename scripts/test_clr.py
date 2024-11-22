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

# numdifftools also installed
import pickle

import simulation_utils


utils.build_metadata_dict()



def plot():
    s_by_s, otu_labels, samples = utils.load_count_data()
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


    clr_s_by_s_dna, clr_s_by_s_rna, occupancy_idx, otu_labels_occupancy = utils.clr_transform(s_by_s, otu_labels, samples, min_occupancy = 1, pseudocount = 1)

    clr_s_by_s_dna, clr_s_by_s_rna, otu_labels_occupancy =  utils.clr_transform_across_samples(s_by_s, otu_labels, samples, min_occupancy = 1)

    clr_s_by_s_dna_subset = clr_s_by_s_dna[occupancy_idx,:]
    clr_s_by_s_rna_subset = clr_s_by_s_rna[occupancy_idx,:]

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels_occupancy)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            ax.scatter(days, clr_s_by_s_rna[c,:], s=8, alpha=1, c=utils.dna_rna_color_dict['RNA'], zorder=1)
            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel("CLR-transformed abund., " + utils.rescaled_label_clr_dict['RNA'], fontsize=10)
            ax.set_title(otu_labels_occupancy[c], fontsize=11)



    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stest_clr.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




plot()

