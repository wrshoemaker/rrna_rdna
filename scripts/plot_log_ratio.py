import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats, signal



s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)


rel_s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
rel_s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)

t = rel_s_by_s_rescaled_rna[:,1:]/rel_s_by_s_rescaled_rna[:,:-1]


#(25, 123)
log_ratio_dna = numpy.log10(rel_s_by_s_rescaled_dna[:,1:]/rel_s_by_s_rescaled_dna[:,:-1])
log_ratio_rna = numpy.log10(rel_s_by_s_rescaled_rna[:,1:]/rel_s_by_s_rescaled_rna[:,:-1])

#print(rel_s_by_s_rescaled_rna[0,:])
fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(bottom= 0.15)

data_type_all = ['DNA', 'RNA']
ratio_label = [r'$\Delta  \ell_{\tilde{x}_{i}^{(d)}}$', r'$\Delta  \ell_{\tilde{x}_{i}^{(r)}}$']
ratio_time_label = [r'$\Delta  \ell_{\tilde{x}_{i}^{(d)}}(t)$', r'$\Delta  \ell_{\tilde{x}_{i}^{(r)}}(t)$']
for log_ratio_i_idx, log_ratio_i in enumerate([log_ratio_dna, log_ratio_rna]):

    ax_dist = plt.subplot2grid((2, 2), (0, log_ratio_i_idx), colspan=1)
    ax_time = plt.subplot2grid((2, 2), (1, log_ratio_i_idx), colspan=1)

    data_type_i = data_type_all[log_ratio_i_idx]

    for afd in log_ratio_dna:

        hist_afd, bins_afd = utils.get_hist_and_bins(afd, bins=12)
        ax_dist.scatter(bins_afd, hist_afd, s=7, color=utils.dna_rna_color_dict[data_type_i], alpha=0.7, lw=1)
        #ax_dist.plot(bins_afd, hist_afd, lw=1, color=utils.dna_rna_color_dict[data_type_i], alpha=0.7)

        
        #ax_dist.hist(afd, bins=12, density=True, histtype='step', alpha=0.5, color=utils.dna_rna_color_dict[data_type_i], zorder=1)
        
        ax_time.plot(days[:-1], afd, ls='-', lw=0.5, alpha=0.3, c=utils.dna_rna_color_dict[data_type_i], zorder=1)
        ax_time.scatter(days[:-1], afd, s=0.8 , c=utils.dna_rna_color_dict[data_type_i], alpha=0.7, zorder=2)


    ax_dist.axvline(x=0, ls=':', c='k', lw=2, zorder=2, label='Stationarity')
    ax_dist.set_yscale('log', basey=10, nonposy='clip')
    ax_dist.set_xlabel("Log ratio of abundances between\nconsecutive timepoints, " + ratio_label[log_ratio_i_idx], fontsize=10)
    ax_dist.set_ylabel("Probability density", fontsize=10)
    ax_dist.set_title(data_type_i, fontsize=12)


    ax_time.axhline(y=0, ls=':', c='k', lw=2, zorder=2, label='Stationarity')
    ax_time.set_xlabel("Time (days), " + r'$t$', fontsize=10)
    ax_time.set_ylabel("Log ratio of abundances between\nconsecutive timepoints at time " + r'$t$' + ', ' + ratio_time_label[log_ratio_i_idx], fontsize=10)
    
    if log_ratio_i_idx == 0:
        ax_dist.legend(loc='upper left', fontsize=8)
        ax_time.legend(loc='upper left', fontsize=8)


fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%slog_ratio.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

