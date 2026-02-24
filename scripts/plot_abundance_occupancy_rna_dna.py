import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats, signal
# numdifftools also installed
import pickle



s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])
delta_days = days[1:] - days[:-1]


fig = plt.figure(figsize = (4.5, 8))
fig.subplots_adjust(bottom= 0.15)


for pair_type_idx, pair_type in enumerate([['DNA', 'RNA'], ['RNA', 'DNA']]):

    #ax = plt.subplot2grid((2, 2), (mean_type_idx, occupancy_type_idx))

    s_by_s_mean = s_by_s[:,(sample_type==pair_type[0])]
    s_by_s_occupancy = s_by_s[:,(sample_type==pair_type[1])]

    occupancies_1, occupancies_2, predicted_occupancies_1, predicted_occupancies_2, mad, beta = utils.predict_occupancy_provide_mean(s_by_s_mean, s_by_s_occupancy)



    idx_to_keep = (occupancies_2>0) & (predicted_occupancies_2 > 0) & (mad > 0)
    mad = mad[idx_to_keep]
    occupancies = occupancies_2[idx_to_keep]
    predicted_occupancies = predicted_occupancies_2[idx_to_keep]

    mad_occupancy_joint = numpy.concatenate((mad, occupancies),axis=0)
    min_ = min(mad_occupancy_joint)
    max_ = max(mad_occupancy_joint)

    sorted_plot_data = utils.plot_color_by_pt_dens(mad, occupancies, radius=utils.color_radius, loglog=1)
    x,y,z = sorted_plot_data[:, 0], sorted_plot_data[:, 1], sorted_plot_data[:, 2]

    ax = plt.subplot2grid((2, 1), (pair_type_idx, 0))
    ax.scatter(x, y, c=numpy.sqrt(z), cmap=utils.cmap_data_type_dict[pair_type[0]], s=70, alpha=0.9, edgecolors='none', zorder=1)
    
    #all_ = numpy.concatenate([x, y])
    # mad vs occupancy
    mad_log10 = numpy.log10(mad)
    occupancies_log10 = numpy.log10(occupancies)
    predicted_occupancies_log10 = numpy.log10(predicted_occupancies)
    hist_all, bin_edges_all = numpy.histogram(mad_log10, density=True, bins=25)
    bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_mean_all_to_keep = []
    bins_occupancies = []
    for i in range(0, len(bin_edges_all)-1 ):
        predicted_occupancies_log10_i = predicted_occupancies_log10[(mad_log10>=bin_edges_all[i]) & (mad_log10<bin_edges_all[i+1])]
        #bins_mean_all_to_keep.append(bins_mean_all[i])
        bins_mean_all_to_keep.append(bin_edges_all[i])
        bins_occupancies.append(numpy.mean(predicted_occupancies_log10_i))


    bins_mean_all_to_keep = numpy.asarray(bins_mean_all_to_keep)
    bins_occupancies = numpy.asarray(bins_occupancies)

    bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_occupancies))]
    bins_occupancies_no_nan = bins_occupancies[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_occupancies))]

    ax.plot(10**bins_mean_all_to_keep_no_nan, 10**bins_occupancies_no_nan, lw=3, ls='--',c='k', zorder=2, label='Gamma prediction')

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    #ax_mad.tick_params(axis='both', which='minor', labelsize=9)
    #ax_mad.tick_params(axis='both', which='major', labelsize=9)

    ax.set_xlabel("Mean relative abundance, " + pair_type[0], fontsize = 10)
    ax.set_ylabel("Occupancy, " + pair_type[1], fontsize = 10)


    to_keep = (occupancies_1>0) & (occupancies_2>0) & (~numpy.isnan(predicted_occupancies_1)) & (~numpy.isnan(predicted_occupancies_2))
    occupancies_1 = occupancies_1[to_keep]
    occupancies_2 = occupancies_2[to_keep]
    predicted_occupancies_1 = predicted_occupancies_1[to_keep]
    predicted_occupancies_2 = predicted_occupancies_2[to_keep]

    error_1 = numpy.absolute(occupancies_1 - predicted_occupancies_1)/occupancies_1
    error_2 = numpy.absolute(occupancies_2 - predicted_occupancies_2)/occupancies_2

    #print(error_1[;])

    print(numpy.mean(error_1 - error_2))



fig.subplots_adjust(hspace=0.3, wspace=0.35)
fig_name = "%sabundance_occupancy_rna_dna.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
