import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

numpy.seterr(divide='ignore', invalid='ignore')


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

occupancy_rna = numpy.sum((rel_s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
occupancy_dna = numpy.sum((rel_s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)


# OTUs present in all RNA and all DNA samples
subset_idx = (occupancy_rna==1) & (occupancy_dna==1)
rel_s_by_s_rna_subset = rel_s_by_s_rna[subset_idx,:]
rel_s_by_s_dna_subset = rel_s_by_s_dna[subset_idx,:]

mad_rna_subset = numpy.mean(rel_s_by_s_rna_subset, axis=1)
mad_dna_subset = numpy.mean(rel_s_by_s_dna_subset, axis=1)

rescaled_rel_s_by_s_rna_subset = (rel_s_by_s_rna_subset.T/mad_rna_subset).T
rescaled_rel_s_by_s_dna_subset = (rel_s_by_s_dna_subset.T/mad_dna_subset).T

rescaled_rel_s_by_s_ratio_subset = rescaled_rel_s_by_s_rna_subset/rescaled_rel_s_by_s_dna_subset
mean_trajectory_ratio_subset = numpy.mean(rescaled_rel_s_by_s_ratio_subset, axis=0)




# lowest occupancy (# samples)^{-1}
n_occupancy = 10
#occupancy_range = numpy.logspace(numpy.log10(1/rel_s_by_s_dna.shape[1]), 0, num=n_occupancy, base=10)
# lower occupancy bound of 0.1
occupancy_range = numpy.logspace(numpy.log10(0.05), 0, num=n_occupancy, base=10)

#  endpoint=True
cmap_offset = int(0.2*16)
rgb_blue_occupancy = cm.Blues(numpy.linspace(0,1,n_occupancy+10))
rgb_blue_occupancy = colors.ListedColormap(rgb_blue_occupancy[cmap_offset:,:-1])




fig = plt.figure(figsize = (4, 4))
fig.subplots_adjust(bottom= 0.15)
ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

ax.plot(days, mean_trajectory_ratio_subset, lw=1, c='k', zorder=1, label='RNA and DNA occupancy ' + r'$= 1$')


for occupancy_i in occupancy_range:

    i_idx = (occupancy_rna<occupancy_i) & (occupancy_dna<occupancy_i)

    n_otus = sum(i_idx)

    if n_otus < 20:
        continue

    rel_s_by_s_rna_i = rel_s_by_s_rna[i_idx,:]
    rel_s_by_s_dna_i = rel_s_by_s_dna[i_idx,:]

    mad_rna_i = numpy.mean(rel_s_by_s_rna_i, axis=1)
    mad_dna_i = numpy.mean(rel_s_by_s_dna_i, axis=1)

    rescaled_rel_s_by_s_rna_i = (rel_s_by_s_rna_i.T/mad_rna_i).T
    rescaled_rel_s_by_s_dna_i = (rel_s_by_s_dna_i.T/mad_dna_i).T

    rescaled_rel_s_by_s_ratio_i = rescaled_rel_s_by_s_rna_i/rescaled_rel_s_by_s_dna_i
    n_otus = rescaled_rel_s_by_s_ratio_i.shape[0]

    # remove OTUs with all zeros or nans
    # identify OTUs where the numbers of observations that (not finite (nan) or (|) equal to zero) is equal to length of timeseries
    to_keep_i_idx = ~(numpy.sum(~numpy.isfinite(rescaled_rel_s_by_s_ratio_i) | (rescaled_rel_s_by_s_ratio_i==0), axis=1) == len(days))

    rescaled_rel_s_by_s_ratio_i = rescaled_rel_s_by_s_ratio_i[to_keep_i_idx,:]

    # sad_t for 
    mean_trajectory_ratio_t_all = []
    days_t_all = []
    for sad_t_idx, sad_t in enumerate(rescaled_rel_s_by_s_ratio_i.T):

        sad_t = sad_t[numpy.isfinite(sad_t)]

        sad_t = sad_t[(sad_t!=0) & (sad_t!=1)]


        if len(sad_t) > 20:
            mean_trajectory_ratio_t_all.append(numpy.mean(sad_t))
            days_t_all.append(days[sad_t_idx])

    print(len(days_t_all))


    ax.scatter(days_t_all, mean_trajectory_ratio_t_all, s=6, color=rgb_blue_occupancy(occupancy_i), linewidth=0.5, edgecolors='k', alpha=0.9, zorder=2)




ax.set_xlabel("Time (days)", fontsize = 10)
ax.set_ylabel("Mean ratio of rescaled RNA\nand DNA relative abundances, " + r'$ \left< \phi_{i} (t) | o_{i} \leq o \right>_{S}$', fontsize = 10)

ax.legend(loc="upper left")



fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%sphi_low_occupancy.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


