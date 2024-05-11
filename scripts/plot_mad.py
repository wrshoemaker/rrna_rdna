import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


#from matplotlib.axes._axes import _log as matplotlib_axes_logger
#matplotlib_axes_logger.setLevel('ERROR')


s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)
# s_by_s.shape = (246, 134265)

metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
sample_type_days = numpy.asarray([metadata_dict[s]['day'] for s in samples])


sample_type_rna_idx = (sample_type=='RNA') & (sample_type_days<=365)
sample_type_dna_idx = (sample_type=='DNA') & (sample_type_days<=365)


sample_type_rna = samples[sample_type_rna_idx]



rel_s_by_s_rna = rel_s_by_s[:,sample_type_rna_idx]
rel_s_by_s_dna = rel_s_by_s[:,sample_type_dna_idx]


mad_rna = numpy.mean(rel_s_by_s_rna, axis=1)
mad_dna = numpy.mean(rel_s_by_s_dna, axis=1)

to_keep_idx = (mad_rna>0) & (mad_dna>0)

occupancy_rna = numpy.sum((rel_s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
occupancy_dna = numpy.sum((rel_s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)

occupancy_one_rna_idx = (occupancy_rna==1)
occupancy_one_dna_idx = (occupancy_dna==1)

mad_rna_to_plot = mad_rna[to_keep_idx]
mad_dna_to_plot = mad_dna[to_keep_idx]

log_mad_rna = numpy.log10(mad_rna_to_plot)
log_mad_dna = numpy.log10(mad_dna_to_plot)

rescaled_log_mad_rna = (log_mad_rna - numpy.mean(log_mad_rna))/numpy.std(log_mad_rna)
rescaled_log_mad_dna = (log_mad_dna - numpy.mean(log_mad_dna))/numpy.std(log_mad_dna)


# occupancy one
mad_dna_occupancy_one = mad_dna[occupancy_one_dna_idx]
mad_rna_occupancy_one = mad_rna[occupancy_one_rna_idx]

log_mad_dna_occupancy_one = numpy.log10(mad_dna_occupancy_one)
log_mad_rna_occupancy_one = numpy.log10(mad_rna_occupancy_one)

rescaled_log_mad_dna_occupancy_one = (log_mad_dna_occupancy_one - numpy.mean(log_mad_dna_occupancy_one))/numpy.std(log_mad_dna_occupancy_one)
rescaled_log_mad_rna_occupancy_one = (log_mad_rna_occupancy_one - numpy.mean(log_mad_rna_occupancy_one))/numpy.std(log_mad_rna_occupancy_one)



fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

ax_mad = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax_mad_corr = plt.subplot2grid((1, 2), (0, 1), colspan=1)



# ax_mad
hist_dna, bins_mean_dna = utils.get_hist_and_bins(rescaled_log_mad_dna, bins=100)
hist_rna, bins_mean_rna = utils.get_hist_and_bins(rescaled_log_mad_rna, bins=100)


ax_mad.scatter(bins_mean_dna, hist_dna, s=7, color='dodgerblue', alpha=0.5, lw=1, label='DNA')
ax_mad.scatter(bins_mean_rna, hist_rna, s=7, color='#FF6347', alpha=0.5, lw=1, label='RNA')



#print(len(rescaled_log_mad_dna_occupancy_one))
hist_dna_occupancy_one, bins_mean_dna_occupancy_one = utils.get_hist_and_bins(rescaled_log_mad_dna_occupancy_one, bins=8)
hist_rna_occupancy_one, bins_mean_rna_occupancy_one = utils.get_hist_and_bins(rescaled_log_mad_rna_occupancy_one, bins=8)

ax_mad.scatter(bins_mean_dna_occupancy_one, hist_dna_occupancy_one, s=7, color='dodgerblue', marker="^", alpha=0.5, lw=1, label='DNA, occupancy=1')
ax_mad.scatter(bins_mean_rna_occupancy_one, hist_rna_occupancy_one, s=7, color='#FF6347', marker="^", alpha=0.5, lw=1, label='RNA, occupancy=1')




ax_mad.set_yscale('log', basey=10)

ax_mad.set_xlabel("Rescaled log mean relative abundance", fontsize = 10)
ax_mad.set_ylabel("Probability density", fontsize = 10)

ax_mad.legend(loc="upper right")


# ax_mad_corr

ax_mad_corr.scatter(mad_dna_to_plot, mad_rna_to_plot, s=2, color='k', alpha=0.3)


min_mad = min(numpy.concatenate([mad_dna, mad_rna]))
ax_mad_corr.plot([min_mad,1],[min_mad,1], lw=2, ls=':',c='k', zorder=2)


ax_mad_corr.set_xlabel("Mean relative abundance, DNA", fontsize = 10)
ax_mad_corr.set_ylabel("Mean relative abundance, RNA", fontsize = 10)


ax_mad_corr.set_xscale('log', basex=10)
ax_mad_corr.set_yscale('log', basey=10)



rho = numpy.corrcoef(log_mad_rna, log_mad_dna)[0,1]

print(rho, rho**2)






fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%smad.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
