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

sample_type_rna_idx = (sample_type=='RNA')
sample_type_dna_idx = (sample_type=='DNA')

sample_type_rna = samples[sample_type_rna_idx]

rel_s_by_s_rna = rel_s_by_s[:,sample_type_rna_idx]
rel_s_by_s_dna = rel_s_by_s[:,sample_type_dna_idx]

days = numpy.asarray([metadata_dict[s]['day'] for s in sample_type_rna])

#occupancy_rna = numpy.sum((rel_s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
#occupancy_dna = numpy.sum((rel_s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)
#subset_idx = (occupancy_rna==1) & (occupancy_dna==1)

#rel_s_by_s_rna_subset = rel_s_by_s_rna[subset_idx,:]
#rel_s_by_s_dna_subset = rel_s_by_s_dna[subset_idx,:]

#richness_rna = numpy.sum(rel_s_by_s_rna_subset>0, axis=0)
#richness_dna = numpy.sum(rel_s_by_s_dna>0, axis=0)

richness_rna = numpy.apply_along_axis(utils.calculate_richness, 0, rel_s_by_s_rna)
richness_dna = numpy.apply_along_axis(utils.calculate_richness, 0, rel_s_by_s_dna)

evenness_rna = numpy.apply_along_axis(utils.calculate_pielou_evenness, 0, rel_s_by_s_rna)
evenness_dna = numpy.apply_along_axis(utils.calculate_pielou_evenness, 0, rel_s_by_s_dna)




fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

ax_richness = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax_evenness = plt.subplot2grid((1, 2), (0, 1), colspan=1)

#ax_evenness = plt.subplot2grid((1, 1), (0, 0), colspan=1)


# ax_richness
ax_richness.plot(days, richness_dna, lw=0.8, ls='-', alpha=1, c='k', zorder=1)
ax_richness.scatter(days, richness_dna, s=5, alpha=0.8, c='k', zorder=1, label='DNA')

ax_richness.plot(days, richness_rna, lw=0.8, ls='-', alpha=1, c='k', zorder=1)
ax_richness.scatter(days, richness_rna, s=20, alpha=0.8, zorder=2, facecolors='white', edgecolors='k', label='RNA')

ax_richness.set_yscale('log', basey=10)

ax_richness.set_xlabel("Time (days)", fontsize = 12)
ax_richness.set_ylabel("Richness", fontsize = 12)


# ax_evenness
ax_evenness.plot(days, evenness_dna, lw=0.8, ls='-', alpha=1, c='k', zorder=1)
ax_evenness.scatter(days, evenness_dna, s=5, alpha=0.8, c='k', zorder=1, label='DNA')

ax_evenness.plot(days, evenness_rna, lw=0.8, ls='-', alpha=1, c='k', zorder=1)
ax_evenness.scatter(days, evenness_rna, s=20, alpha=0.8, zorder=2, facecolors='white', edgecolors='k', label='RNA')

ax_evenness.legend(loc="lower right")

ax_evenness.set_xlabel("Time (days)", fontsize = 12)
ax_evenness.set_ylabel("Evenness", fontsize = 12)


fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%stime_vs_richness.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

