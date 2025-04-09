import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy import stats


numpy.seterr(divide='ignore', invalid='ignore')
min_n_obs = 10



s_by_s, otu_labels, samples = utils.load_count_data()
# s_by_s.shape = (246, 134265)

s_by_s_dna_all, s_by_s_rna_all = utils.subset_s_by_s_occupancy(s_by_s, samples, min_occupancy=0.1)
s_by_s_dna_subset, s_by_s_rna_subset = utils.subset_s_by_s_occupancy(s_by_s, samples, min_occupancy=1)

rel_s_by_s_dna_all = s_by_s_dna_all/numpy.sum(s_by_s_dna_all, axis=0)
rel_s_by_s_rna_all = s_by_s_rna_all/numpy.sum(s_by_s_rna_all, axis=0)

rel_s_by_s_dna_subset = s_by_s_dna_subset/numpy.sum(s_by_s_dna_subset, axis=0)
rel_s_by_s_rna_subset = s_by_s_rna_subset/numpy.sum(s_by_s_rna_subset, axis=0)

mad_dna_all = numpy.mean(rel_s_by_s_dna_all, axis=1)
mad_rna_all = numpy.mean(rel_s_by_s_rna_all, axis=1)
mad_dna_subset = numpy.mean(rel_s_by_s_dna_subset, axis=1)
mad_rna_subset = numpy.mean(rel_s_by_s_rna_subset, axis=1)

mad_dna_all = numpy.sort(mad_dna_all[mad_dna_all>0])[::-1]
mad_rna_all = numpy.sort(mad_rna_all[mad_rna_all>0])[::-1]
mad_dna_subset = numpy.sort(mad_dna_subset[mad_dna_subset>0])[::-1]
mad_rna_subset = numpy.sort(mad_rna_subset[mad_rna_subset>0])[::-1]



fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

ax_all = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax_subset = plt.subplot2grid((1, 2), (0, 1), colspan=1)


ax_all.plot(range(len(mad_dna_all)), mad_dna_all, lw=1, alpha=0.9, c=utils.dna_rna_color_dict['DNA'], label='DNA')
ax_all.plot(range(len(mad_rna_all)), mad_rna_all, lw=1, alpha=0.9, c=utils.dna_rna_color_dict['RNA'], label='RNA')
ax_all.set_yscale('log', basey=10)

ax_all.legend(loc='upper right')

ax_subset.scatter(range(len(mad_dna_subset)), mad_dna_subset, s=7, alpha=1, c=utils.dna_rna_color_dict['DNA'], zorder=2)
ax_subset.scatter(range(len(mad_rna_subset)), mad_rna_subset, s=7, alpha=1, c=utils.dna_rna_color_dict['RNA'], zorder=2)

ax_subset.plot(range(len(mad_dna_subset)), mad_dna_subset, lw=1, alpha=0.9, c=utils.dna_rna_color_dict['DNA'], zorder=1)
ax_subset.plot(range(len(mad_rna_subset)), mad_rna_subset, lw=1, alpha=0.9, c=utils.dna_rna_color_dict['RNA'], zorder=1)
ax_subset.set_yscale('log', basey=10)

ax_all.set_title('Occupancy ' + r'$\geq 0.1$')
ax_subset.set_title('Occupancy ' + r'$= 1$')


ax_all.set_xlabel("Rank", fontsize=10)
ax_subset.set_xlabel("Rank", fontsize=10)

ax_all.set_ylabel("Mean relative abundance", fontsize=10)
ax_subset.set_ylabel("Mean relative abundance", fontsize=10)



fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%smad_rad.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

