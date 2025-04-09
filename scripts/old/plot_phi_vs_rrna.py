import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy import stats


taxonomic_level = 'genus'
taxonomy_dict = utils.build_taxonomy_dict()

samples_dna_copy_number_dna, days_copy_number_dna, coarse_labels_dna, mean_copy_number_dna = utils.calculate_mean_copy_number('DNA', taxonomic_level)
samples_dna_copy_number_rna, days_copy_number_rna, coarse_labels_rna, mean_copy_number_rna = utils.calculate_mean_copy_number('RNA', taxonomic_level)



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
water_temp = numpy.asarray([metadata_dict[s]['water_temp'] for s in sample_type_rna])
water_temp_non_nan_idx = ~numpy.isnan(water_temp)
water_temp_non_nan = water_temp[water_temp_non_nan_idx]


fig = plt.figure(figsize = (4, 4))
fig.subplots_adjust(bottom= 0.15)

ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)


# get phi trajectory for RNA and DNA using OTUs used to calculate copy number...
# [DNA, RNA]
coarse_labels_set_all = [('DNA', coarse_labels_dna, mean_copy_number_dna), ('RNA', coarse_labels_rna, mean_copy_number_rna)]
for coarse_labels_set_i in coarse_labels_set_all:

    type_i, coarse_labels_i, mean_copy_number_i = coarse_labels_set_i

    otu_to_keep_i = []
    for key_i, value_i in taxonomy_dict.items():

        if value_i['genus'] in coarse_labels_i:
            otu_to_keep_i.append(key_i)

    otu_to_keep_i = numpy.unique(otu_to_keep_i)

    otu_to_keep_i_idx = numpy.asarray([numpy.where(otu_labels==j)[0][0] for j in otu_to_keep_i])


    rel_s_by_s_rna_i = rel_s_by_s_rna[otu_to_keep_i_idx,:]
    rel_s_by_s_dna_i = rel_s_by_s_dna[otu_to_keep_i_idx,:]

    occupancy_rna_i = numpy.sum((rel_s_by_s_rna_i>0), axis=1)/sum(sample_type_rna_idx)
    occupancy_dna_i = numpy.sum((rel_s_by_s_dna_i>0), axis=1)/sum(sample_type_dna_idx)

    subset_i_idx = (occupancy_rna_i==1) & (occupancy_dna_i==1)

    rel_s_by_s_rna_i_subset = rel_s_by_s_rna_i[subset_i_idx,:]
    rel_s_by_s_dna_i_subset = rel_s_by_s_dna_i[subset_i_idx,:]

    mad_rna_i_subset = numpy.mean(rel_s_by_s_rna_i_subset, axis=1)
    mad_dna_i_subset = numpy.mean(rel_s_by_s_dna_i_subset, axis=1)

    rescaled_rel_s_by_s_rna_i_subset = (rel_s_by_s_rna_i_subset.T/mad_rna_i_subset).T
    rescaled_rel_s_by_s_dna_i_subset = (rel_s_by_s_dna_i_subset.T/mad_dna_i_subset).T

    rescaled_rel_s_by_s_ratio_i_subset = rescaled_rel_s_by_s_rna_i_subset/rescaled_rel_s_by_s_dna_i_subset
    mean_trajectory_ratio_i_subset = numpy.mean(rescaled_rel_s_by_s_dna_i_subset, axis=0)


    slope, intercept, r_value, p_value, std_err = stats.linregress(mean_trajectory_ratio_i_subset, mean_copy_number_i)
    merged_ = numpy.concatenate([mean_trajectory_ratio_i_subset, mean_copy_number_i])
    x_range_ =  numpy.linspace(min(merged_) , max(merged_) , 10000)
    y_fit_range = slope*x_range_ + intercept


    ax.scatter(mean_trajectory_ratio_i_subset, mean_copy_number_i, s=5, alpha=0.8, c=utils.dna_rna_color_dict[type_i], zorder=1, label=type_i)
    ax.plot(x_range_, y_fit_range, lw=1.5, ls='--', zorder=1, c=utils.dna_rna_color_dict[type_i])



    


ax.legend(loc="upper left")

ax.set_xlabel("Mean ratio of rescaled RNA\nand DNA relative abundances, " + r'$ \left< \phi_{i} (t) | o_{i} \leq o \right>_{S}$', fontsize = 10)
#ax.set_xlabel("Mean ratio of rescaled RNA\nand DNA relative abundances, " + r'$ \left< \phi_{i} (t) | o_{i} \leq o \right>_{S}$', fontsize = 10)

ax.set_ylabel("Mean rRNA copy number", fontsize = 10)


fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%sphi_vs_rrna.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



