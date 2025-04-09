import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


#from matplotlib.axes._axes import _log as matplotlib_axes_logger
#matplotlib_axes_logger.setLevel('ERROR')


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

occupancy_nonzero_both_idx = (occupancy_rna==1) & (occupancy_dna==1)
rel_s_by_s_rna_both = rel_s_by_s_rna[occupancy_nonzero_both_idx,:]
rel_s_by_s_dna_both = rel_s_by_s_dna[occupancy_nonzero_both_idx,:]
#occupancy_dna_filtered_both = occupancy_dna[occupancy_nonzero_both_idx]

rel_s_by_s_rna_dna = rel_s_by_s_rna[(occupancy_dna==1),:]
rel_s_by_s_dna_dna = rel_s_by_s_dna[(occupancy_dna==1),:]

mean_ratio_both_all = numpy.mean(rel_s_by_s_rna_both/rel_s_by_s_dna_both, axis=1)
mean_ratio_dna_all = numpy.mean(rel_s_by_s_rna_dna/rel_s_by_s_dna_dna, axis=1)

ratio_mean_both_all = numpy.mean(rel_s_by_s_rna_both, axis=1)/numpy.mean(rel_s_by_s_dna_both, axis=1)
ratio_mean_dna_all = numpy.mean(rel_s_by_s_rna_dna, axis=1)/numpy.mean(rel_s_by_s_dna_dna, axis=1)




# rescale
mean_ratio_both_all_log = numpy.log10(mean_ratio_both_all)



# plot
fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(bottom= 0.15)

ax_mad_dna = plt.subplot2grid((2, 2), (0, 0), colspan=1)
ax_rho_dna = plt.subplot2grid((2, 2), (0, 1), colspan=1)
ax_mad_both = plt.subplot2grid((2, 2), (1, 0), colspan=1)
ax_rho_both = plt.subplot2grid((2, 2), (1, 1), colspan=1)


# ax_mad_dna
hist_mean_ratio_dna, bins_mean_mean_ratio_dna = utils.get_hist_and_bins(utils.rescale_log(mean_ratio_dna_all), bins=8)
hist_ratio_mean_dna, bins_mean_ratio_mean_dna = utils.get_hist_and_bins(utils.rescale_log(ratio_mean_dna_all), bins=8)

ax_mad_dna.scatter(bins_mean_mean_ratio_dna, hist_mean_ratio_dna, s=7, color='dodgerblue', alpha=1, lw=1, label='Mean of ratio')
ax_mad_dna.scatter(bins_mean_ratio_mean_dna, hist_ratio_mean_dna, s=7, color='#FF6347', alpha=1, lw=1, label='Ratio of means')

ax_mad_dna.plot(bins_mean_mean_ratio_dna, hist_mean_ratio_dna, lw=1, color='dodgerblue', alpha=0.8)
ax_mad_dna.plot(bins_mean_ratio_mean_dna, hist_ratio_mean_dna, lw=1, color='#FF6347', alpha=0.8)

ax_mad_dna.set_xlabel("Rescaled log", fontsize = 12)
ax_mad_dna.set_ylabel("Probability density", fontsize = 12)

ax_mad_dna.set_yscale('log', basey=10)

ax_mad_dna.legend(loc="upper right", fontsize=6)


# ax_rho_dna
ax_rho_dna.scatter(mean_ratio_dna_all, ratio_mean_dna_all, s=10, color='k', alpha=0.8, zorder=2)
min_mad_dna = min(numpy.concatenate([mean_ratio_dna_all, ratio_mean_dna_all]))
max_mad_dna = max(numpy.concatenate([mean_ratio_dna_all, ratio_mean_dna_all]))
ax_rho_dna.plot([min_mad_dna,max_mad_dna],[min_mad_dna,max_mad_dna], lw=2, ls=':',c='k', zorder=1)
ax_rho_dna.set_xlim([min_mad_dna, max_mad_dna])
ax_rho_dna.set_ylim([min_mad_dna, max_mad_dna])


ax_rho_dna.set_xlabel("Mean of RNA/DNA ratio", fontsize = 10)
ax_rho_dna.set_ylabel("Ratio of RNA and DNA means", fontsize = 10)

ax_rho_dna.set_xscale('log', basex=10)
ax_rho_dna.set_yscale('log', basey=10)

rho_dna = numpy.corrcoef(numpy.log10(mean_ratio_dna_all), numpy.log10(ratio_mean_dna_all))[0,1]




# ax_mad_both
hist_mean_ratio_both, bins_mean_mean_ratio_both = utils.get_hist_and_bins(utils.rescale_log(mean_ratio_both_all), bins=8)
hist_ratio_mean_both, bins_mean_ratio_mean_both = utils.get_hist_and_bins(utils.rescale_log(ratio_mean_both_all), bins=8)

ax_mad_both.scatter(bins_mean_mean_ratio_both, hist_mean_ratio_both, s=7, color='dodgerblue', alpha=0.5, lw=1, label='DNA')
ax_mad_both.scatter(bins_mean_ratio_mean_both, hist_ratio_mean_both, s=7, color='#FF6347', alpha=0.5, lw=1, label='RNA')

ax_mad_both.plot(bins_mean_mean_ratio_both, hist_mean_ratio_both, lw=1, color='dodgerblue', alpha=0.8)
ax_mad_both.plot(bins_mean_ratio_mean_both, hist_ratio_mean_both, lw=1, color='#FF6347', alpha=0.8)

ax_mad_both.set_yscale('log', basey=10)

ax_mad_both.set_xlabel("Rescaled log", fontsize = 12)
ax_mad_both.set_ylabel("Probability density", fontsize = 12)





# ax_rho_both
ax_rho_both.scatter(mean_ratio_both_all, ratio_mean_both_all, s=10, color='k', alpha=0.8, zorder=2)
min_mad_both = min(numpy.concatenate([mean_ratio_both_all, ratio_mean_both_all]))
max_mad_both = max(numpy.concatenate([mean_ratio_both_all, ratio_mean_both_all]))
ax_rho_both.plot([min_mad_both,max_mad_both],[min_mad_both,max_mad_both], lw=2, ls=':',c='k', zorder=1)
ax_rho_both.set_xlim([min_mad_both, max_mad_both])
ax_rho_both.set_ylim([min_mad_both, max_mad_both])


ax_rho_both.set_xlabel("Mean of RNA/DNA ratio", fontsize = 10)
ax_rho_both.set_ylabel("Ratio of RNA and DNA means", fontsize = 10)

ax_rho_both.set_xscale('log', basex=10)
ax_rho_both.set_yscale('log', basey=10)

rho_both = numpy.corrcoef(numpy.log10(mean_ratio_both_all), numpy.log10(ratio_mean_both_all))[0,1]





fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%smad_ratio.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()




# plot ratio of averages vs. average dna
fig = plt.figure(figsize = (4, 4))
fig.subplots_adjust(bottom= 0.15)

ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

mad_dna = numpy.mean(rel_s_by_s_dna_dna, axis=1)

ax.scatter(mad_dna, ratio_mean_dna_all, s=10, color='k', alpha=0.8, zorder=2)
min_mad_both = min(numpy.concatenate([mad_dna, ratio_mean_dna_all]))
max_mad_both = max(numpy.concatenate([mad_dna, ratio_mean_dna_all]))
ax.plot([min_mad_both,max_mad_both],[min_mad_both,max_mad_both], lw=2, ls=':',c='k', zorder=1)
ax.set_xlim([min_mad_both, max_mad_both])
ax.set_ylim([min_mad_both, max_mad_both])

ax.set_xlabel("Mean relative abundance, DNA", fontsize = 10)
ax.set_ylabel("Ratio of RNA and DNA means", fontsize = 10)

ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)


rho_both = numpy.corrcoef(numpy.log10(ratio_mean_dna_all), numpy.log10(mad_dna))[0,1]
print(rho_both**2)


fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%smad_ratio_vs_dna.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



