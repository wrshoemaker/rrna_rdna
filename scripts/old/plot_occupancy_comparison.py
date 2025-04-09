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


occupancy_nonzero_idx = (occupancy_rna>0) & (occupancy_dna>0)
occupancy_rna_filtered = occupancy_rna[occupancy_nonzero_idx]
occupancy_dna_filtered = occupancy_dna[occupancy_nonzero_idx]


min_occupancy = min(numpy.concatenate([occupancy_dna_filtered, occupancy_rna_filtered]))


# fraction detected and absent in each.
prob_detected_rna_absent_dna = sum((occupancy_rna>0) & (occupancy_dna==0))/len(occupancy_rna)
prob_detected_dna_absent_rna = sum((occupancy_dna>0) & (occupancy_rna==0))/len(occupancy_rna)

print(prob_detected_rna_absent_dna, prob_detected_dna_absent_rna)


fig = plt.figure(figsize = (4, 4))
fig.subplots_adjust(bottom= 0.15)

ax_occupancy = plt.subplot2grid((1, 1), (0, 0), colspan=1)
#ax_rna = plt.subplot2grid((1, 2), (0, 1), colspan=1)

ax_occupancy.scatter(occupancy_dna_filtered, occupancy_rna_filtered, s=3, color='dodgerblue', alpha=0.2, lw=1, zorder=2)
ax_occupancy.plot([min_occupancy,1],[min_occupancy,1], lw=2, ls=':',c='k',zorder=1)

ax_occupancy.set_xlim([min_occupancy,1])
ax_occupancy.set_ylim([min_occupancy,1])

ax_occupancy.set_xscale('log', basex=10)
ax_occupancy.set_yscale('log', basey=10)

ax_occupancy.set_xlabel("Occupancy, DNA", fontsize = 10)
ax_occupancy.set_ylabel("Occupancy, RNA", fontsize = 10)




fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%soccupancy_comparison.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()