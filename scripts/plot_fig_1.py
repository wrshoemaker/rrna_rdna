import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy import stats, signal
# numdifftools also installed
import pickle


min_occupany_afd = 1


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])
delta_days = days[1:] - days[:-1]
minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


otu_idx = numpy.where(otu_labels == 'Otu000001')[0][0]


fig = plt.figure(figsize = (4.5, 8))
fig.subplots_adjust(bottom= 0.15)




for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

    sample_type_idx = (sample_type==data_type)
    s_by_s_data = s_by_s[:,sample_type_idx]
    n_reads = numpy.sum(s_by_s_data, axis=0)
    rel_s_by_s_data = s_by_s_data/n_reads

    afd = rel_s_by_s_data[otu_idx,:]
    afd_log10 = numpy.log10(afd)
    logfold = (afd_log10[1:] - afd_log10[:-1])/delta_days


    #rel_s_by_s_data_afd_idx = (numpy.sum(rel_s_by_s_data>0, axis=1)/len(n_reads)) >= min_occupany_afd
    #rel_s_by_s_data_afd = rel_s_by_s_data[rel_s_by_s_data_afd_idx,:]

    ax_timeseries = plt.subplot2grid((2, 1), (data_type_idx, 0))
    #ax_logfold = plt.subplot2grid((2, 2), (data_type_idx, 1))

    if data_type_idx == 0:
        labels = ['b', 'd']
    else:
        labels = ['c', 'e']


    #ax_timeseries.text(-0.1, 1.04, labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_timeseries.transAxes)
    #ax_logfold.text(-0.1, 1.04, labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_logfold.transAxes)



    ax_timeseries.plot(days, afd, lw=1, alpha=1, color=utils.dna_rna_color_dict[data_type], zorder=1)
    ax_timeseries.scatter(days, afd, s=6, alpha=1, color=utils.dna_rna_color_dict[data_type], zorder=1)
    ax_timeseries.set_ylabel("Relative abundance, " + r'$\hat{x}_{i}(t)$' , fontsize=18)
    ax_timeseries.set_yscale('log', basey=10)

    ax_timeseries.set_xlim([0, max(days)])
    ax_timeseries.set_xticks(minor_days, minor=True)
    ax_timeseries.set_xticks(major_days, minor=False)
    ax_timeseries.set_xticklabels(major_labels, minor=False, fontsize=12)
    
    ax_timeseries.yaxis.set_tick_params(labelsize=12)

    if data_type_idx == 1:
        ax_timeseries.set_xlabel("Time (days)", fontsize=18)

    #ax_logfold.plot(days[:-1], logfold, lw=1, alpha=1, color=utils.dna_rna_color_dict[data_type], zorder=2)
    #ax_logfold.scatter(days[:-1], logfold, s=6, alpha=1, color=utils.dna_rna_color_dict[data_type], zorder=2)

    #max_logfold = 1.4*numpy.max(numpy.absolute(logfold))

    #ax_logfold.set_xlim([0, max(days)])
    #ax_logfold.set_ylim([-1*max_logfold, max_logfold])
    #ax_logfold.set_xticks(minor_days, minor=True)
    #ax_logfold.set_xticks(major_days, minor=False)
    #ax_logfold.set_xticklabels(major_labels, minor=False, fontsize=7)
    #ax_logfold.set_xlabel("Time (days)", fontsize=14)
    #ax_logfold.yaxis.set_tick_params(labelsize=7)
    #ax_logfold.set_ylabel("Log-fold change in " + r'$\hat{x}_{i}(t)$' +  " b/w timepoints", fontsize=12)
    #ax_logfold.axhline(y=0, lw=2.5, ls=':', label='Stationarity', color='k', zorder=1)


    #ax_logfold_inset = inset_axes(ax_logfold, width="100%", height="100%", bbox_to_anchor=(0.6,0.1,0.35,0.35), bbox_transform=ax_logfold.transAxes, loc='upper left')
    #ax_logfold_inset.tick_params(labelleft=False, labelbottom=True)
    #ax_logfold_inset.xaxis.set_tick_params(labelsize=6)
    #ax_logfold_inset.hist(logfold, lw=2, alpha=1, bins= 15, color=utils.dna_rna_color_dict[data_type], histtype='step', density=True, zorder=2)
    #ax_logfold_inset.axvline(x=0, lw=2.5, ls=':', color='k', zorder=1)
    #ax_logfold_inset.set_xlim([-1*max_logfold, max_logfold])

    #ax_logfold_inset.set_xlabel("Log-fold abundance ratio, " + r'$\Delta \ell_{x}$', fontsize=8)
    #ax_logfold_inset.set_ylabel("Probability density", fontsize=7)


    #if data_type_idx == 0:
    #    ax_logfold.legend(loc='upper left', fontsize=10)

    if data_type_idx == 0:
        ax_timeseries.set_title('OTU 1 ('+ r'$\mathit{Anabaena}$' + ' sp.)', color='k', fontsize=18)







fig.subplots_adjust(hspace=0.35, wspace=0.25)
fig_name = "%sfig1.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

