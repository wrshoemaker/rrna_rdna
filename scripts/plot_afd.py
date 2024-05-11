import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)


metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

sample_type_rna_idx = (sample_type=='RNA')
sample_type_dna_idx = (sample_type=='DNA')


sample_type_rna = samples[sample_type_rna_idx]

rel_s_by_s_rna = rel_s_by_s[:,sample_type_rna_idx]
rel_s_by_s_dna = rel_s_by_s[:,sample_type_dna_idx]


occupancy_rna = numpy.sum(rel_s_by_s_rna>0, axis=1)/rel_s_by_s_rna.shape[1]
occupancy_dna = numpy.sum(rel_s_by_s_dna>0, axis=1)/rel_s_by_s_dna.shape[1]

rel_s_by_s_rna_final = rel_s_by_s_rna[(occupancy_rna==1),:]
rel_s_by_s_dna_final = rel_s_by_s_dna[(occupancy_dna==1),:]



subset_ratio_idx = (occupancy_rna==1) & (occupancy_dna==1)
rel_s_by_s_rna_ratio = rel_s_by_s_rna[subset_ratio_idx,:]
rel_s_by_s_dna_ratio = rel_s_by_s_dna[subset_ratio_idx,:]



fig = plt.figure(figsize = (12, 4))
fig.subplots_adjust(bottom= 0.15)


ax_dna = plt.subplot2grid((1, 3), (0, 0), colspan=1)
ax_rna = plt.subplot2grid((1, 3), (0, 1), colspan=1)
ax_ratio = plt.subplot2grid((1, 3), (0, 2), colspan=1)



for afd_dna in rel_s_by_s_dna_final:

    log_afd_dna = numpy.log10(afd_dna)
    rescaled_log_afd_dna = (log_afd_dna - numpy.mean(log_afd_dna))/numpy.std(log_afd_dna)

    hist_to_plot, bins_mean_to_plot = utils.get_hist_and_bins(rescaled_log_afd_dna, bins=15)
    ax_dna.scatter(bins_mean_to_plot, hist_to_plot, s=7, color='k', alpha=0.5, lw=1)
    #ax_dna.plot(bins_mean_to_plot, hist_to_plot, lw=0.5, color='k', alpha=0.3, zorder=1)


for afd_rna in rel_s_by_s_rna_final:

    log_afd_rna = numpy.log10(afd_rna)
    rescaled_log_afd_rna = (log_afd_rna - numpy.mean(log_afd_rna))/numpy.std(log_afd_rna)

    hist_to_plot, bins_mean_to_plot = utils.get_hist_and_bins(rescaled_log_afd_rna, bins=15)
    ax_rna.scatter(bins_mean_to_plot, hist_to_plot, s=7, color='k', alpha=0.5, lw=1)


for afd_ratio_idx in range(rel_s_by_s_rna_ratio.shape[0]):

    afd_ratio = rel_s_by_s_rna_ratio[afd_ratio_idx,:]/rel_s_by_s_dna_ratio[afd_ratio_idx,:]

    log_afd_ratio = numpy.log10(afd_ratio)
    rescaled_log_afd_ratio = (log_afd_ratio - numpy.mean(log_afd_ratio))/numpy.std(log_afd_ratio)

    hist_to_plot, bins_mean_to_plot = utils.get_hist_and_bins(rescaled_log_afd_ratio, bins=15)
    ax_ratio.scatter(bins_mean_to_plot, hist_to_plot, s=7, color='k', alpha=0.5, lw=1)




ax_dna.set_yscale('log', basey=10)
ax_rna.set_yscale('log', basey=10)
ax_ratio.set_yscale('log', basey=10)


ax_dna.set_xlabel("Rescaled log relative abundance, DNA", fontsize = 10)
ax_rna.set_xlabel("Rescaled log relative abundance, RNA", fontsize = 10)
ax_ratio.set_xlabel("Rescaled log RNA/DNA\nrelative abundance ratio", fontsize = 10)


ax_dna.set_ylabel("Probability density", fontsize = 10)
ax_rna.set_ylabel("Probability density", fontsize = 10)
ax_ratio.set_ylabel("Probability density", fontsize = 10)





fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%safd.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


#    print(len(afd_dna), sum(sample_type_rna_idx), len(otu_labels))