import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy import stats


taxonomic_level = 'genus'
taxonomy_dict = utils.build_taxonomy_dict()


sample_types_all = ['RNA', 'DNA']


s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)
# s_by_s.shape = (246, 134265)

metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])


fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

for i_idx, i in enumerate(sample_types_all):

    sample_type_i_idx = (sample_type==i)
    sample_type_i = samples[sample_type_i_idx]
    rel_s_by_s_i = rel_s_by_s[:,sample_type_i_idx]

    samples_dna_copy_number_i, days_copy_number_i, coarse_labels_i, mean_copy_number_i = utils.calculate_mean_copy_number(i, taxonomic_level)

    otu_to_keep_i = []
    for key_i, value_i in taxonomy_dict.items():

        if value_i['genus'] in coarse_labels_i:
            otu_to_keep_i.append(key_i)

    otu_to_keep_i = numpy.unique(otu_to_keep_i)

    otu_to_keep_i_idx = numpy.asarray([numpy.where(otu_labels==j)[0][0] for j in otu_to_keep_i])

    rel_s_by_s_i = rel_s_by_s_i[otu_to_keep_i_idx,:]

    mean_across_otus_i = numpy.mean(rel_s_by_s_i, axis=0)

    print(len(mean_across_otus_i))

    ax = plt.subplot2grid((1, 2), (0, i_idx), colspan=1)

    ax.scatter(mean_across_otus_i, mean_copy_number_i, s=5, alpha=0.8, c=utils.dna_rna_color_dict[i], zorder=1, label=i)

    ax.set_xlabel("Mean relative abundance\nacross OTUs at time t, " + i, fontsize = 10)

    ax.set_ylabel("Mean rRNA copy number", fontsize = 10)


ax.legend(loc="upper left")



fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%smad_vs_rrna.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

