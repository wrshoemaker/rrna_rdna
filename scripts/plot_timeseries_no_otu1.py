import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


otu_labels_to_keep = otu_labels!='Otu000001'
otu_labels_no_otu1 = otu_labels[otu_labels_to_keep]
s_by_s_no_otu1 = s_by_s[otu_labels_to_keep, :]

s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)
s_by_s_no_otu1_dna, s_by_s_no_otu1_rna, otu_labels_no_otu1_subset = utils.subset_s_by_s_occupancy(s_by_s_no_otu1, otu_labels_no_otu1, samples, min_occupancy=1)

#print(numpy.mean(s_by_s_no_otu1_dna, axis=1))
#print(otu_labels_no_otu1_subset)

#print(otu_labels_no_otu1_subset)
s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)
s_by_s_rescaled_ratio = s_by_s_rescaled_rna/s_by_s_rescaled_dna

s_by_s_no_otu1_rescaled_dna = utils.rescale_s_by_s(s_by_s_no_otu1_dna)
s_by_s_no_otu1_rescaled_rna = utils.rescale_s_by_s(s_by_s_no_otu1_rna)
s_by_s_no_otu1_rescaled_ratio = s_by_s_no_otu1_rescaled_rna/s_by_s_no_otu1_rescaled_dna

#print(s_by_s_no_otu1_rescaled_dna)


fig = plt.figure(figsize = (12, 8))
ax_dna = plt.subplot2grid((2, 3), (0, 0))
ax_rna = plt.subplot2grid((2, 3), (0, 1))
ax_ratio = plt.subplot2grid((2, 3), (0, 2))

ax_no_otu1_dna = plt.subplot2grid((2, 3), (1, 0))
ax_no_otu1_rna = plt.subplot2grid((2, 3), (1, 1))
ax_no_otu1_ratio = plt.subplot2grid((2, 3), (1, 2))


otu_to_plot = 'Otu000002'
otu_to_plot_idx = numpy.where(otu_labels_subset==otu_to_plot)[0][0]
#print(numpy.where(otu_labels_no_otu1_subset==otu_to_plot))
otu_no_otu1_to_plot_idx = numpy.where(otu_labels_no_otu1_subset==otu_to_plot)[0][0]


ax_dna.plot(days, s_by_s_rescaled_dna[otu_to_plot_idx,:], c=utils.dna_rna_color_dict['DNA'])
ax_rna.plot(days, s_by_s_rescaled_rna[otu_to_plot_idx,:], c=utils.dna_rna_color_dict['RNA'])
ax_ratio.plot(days, s_by_s_rescaled_ratio[otu_to_plot_idx,:], c=utils.dna_rna_color_dict['ratio'])


ax_no_otu1_dna.plot(days, s_by_s_no_otu1_rescaled_dna[otu_no_otu1_to_plot_idx,:], c=utils.dna_rna_color_dict['DNA'])
ax_no_otu1_rna.plot(days, s_by_s_no_otu1_rescaled_rna[otu_no_otu1_to_plot_idx,:], c=utils.dna_rna_color_dict['RNA'])
ax_no_otu1_ratio.plot(days, s_by_s_no_otu1_rescaled_ratio[otu_no_otu1_to_plot_idx,:], c=utils.dna_rna_color_dict['ratio'])


ax_dna.set_yscale('log', basey=10)
ax_dna.set_xlabel("Days", fontsize=10)
ax_dna.set_ylabel("Rescaled relative abundance", fontsize=10)
ax_dna.set_title('Full dataset, DNA', fontsize=12)

ax_rna.set_yscale('log', basey=10)
ax_rna.set_xlabel("Days", fontsize=10)
ax_rna.set_ylabel("Rescaled relative abundance", fontsize=10)
ax_rna.set_title('Full dataset, RNA', fontsize=12)

ax_ratio.set_yscale('log', basey=10)
ax_ratio.set_xlabel("Days", fontsize=10)
ax_ratio.set_ylabel("Rescaled relative abundance", fontsize=10)
ax_ratio.set_title('Full dataset, RNA:DNA ratio', fontsize=12)


ax_no_otu1_dna.set_yscale('log', basey=10)
ax_no_otu1_dna.set_xlabel("Days", fontsize=10)
ax_no_otu1_dna.set_ylabel("Rescaled relative abundance", fontsize=10)
ax_no_otu1_dna.set_title('Excluding OTU1, DNA', fontsize=12)

ax_no_otu1_rna.set_yscale('log', basey=10)
ax_no_otu1_rna.set_xlabel("Days", fontsize=10)
ax_no_otu1_rna.set_ylabel("Rescaled relative abundance", fontsize=10)
ax_no_otu1_rna.set_title('Excluding OTU1, RNA', fontsize=12)

ax_no_otu1_ratio.set_yscale('log', basey=10)
ax_no_otu1_ratio.set_xlabel("Days", fontsize=10)
ax_no_otu1_ratio.set_ylabel("Rescaled relative abundance", fontsize=10)
ax_no_otu1_ratio.set_title('Excluding OTU1, RNA:DNA ratio', fontsize=12)


fig.subplots_adjust(hspace=0.35,wspace=0.4)
fig_name = "%stimeseries_no_otu1.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


#print(numpy.mean(s_by_s_no_otu1_rescaled_dna, axis=1))




