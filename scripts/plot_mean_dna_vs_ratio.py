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


mad_rna = numpy.mean(rel_s_by_s_rna, axis=1)
mad_dna = numpy.mean(rel_s_by_s_dna, axis=1)

ratio_mad = mad_rna/mad_dna

nonzero_ratio_idx = (ratio_mad>0) & (numpy.isfinite(ratio_mad))


mad_dna = mad_dna[nonzero_ratio_idx]
ratio_mad = ratio_mad[nonzero_ratio_idx]

print(len(ratio_mad))



fig = plt.figure(figsize = (4, 4))
fig.subplots_adjust(bottom= 0.15)

ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

ax.scatter(mad_dna, ratio_mad, s=3, color='k', alpha=0.2, zorder=2)
min_mad_both = min(numpy.concatenate([mad_dna, ratio_mad]))
max_mad_both = max(numpy.concatenate([mad_dna, ratio_mad]))
ax.plot([min_mad_both,max_mad_both],[min_mad_both,max_mad_both], lw=2, ls=':',c='k', zorder=1)
ax.set_xlim([min_mad_both, max_mad_both])
ax.set_ylim([min_mad_both, max_mad_both])

ax.set_xlabel("Mean relative abundance, DNA", fontsize = 10)
ax.set_ylabel("Ratio of RNA and DNA means", fontsize = 10)

ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)


rho_both = numpy.corrcoef(numpy.log10(ratio_mad), numpy.log10(mad_dna))[0,1]
print(rho_both**2)


fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%smean_dna_vs_ratio_of_means.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
