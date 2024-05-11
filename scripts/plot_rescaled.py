import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')



numpy.random.seed(123456789)


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


subset_idx = (occupancy_rna==1) & (occupancy_dna==1)

rel_s_by_s_rna_subset = rel_s_by_s_rna[subset_idx,:]
rel_s_by_s_dna_subset = rel_s_by_s_dna[subset_idx,:]


afd_ratio_i_all = []
afd_rna_i_all = []
afd_dna_i_all = []
for i_idx in range(sum(subset_idx)):

    afd_rna_i = rel_s_by_s_rna_subset[i_idx,:]
    afd_dna_i = rel_s_by_s_dna_subset[i_idx,:]

    afd_ratio_i = afd_rna_i/afd_dna_i
    afd_ratio_i_all.append(afd_ratio_i)
    afd_rna_i_all.append(afd_rna_i)
    afd_dna_i_all.append(afd_dna_i)





fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(bottom= 0.15)

ax_dna = plt.subplot2grid((3, 1), (0, 0), colspan=1)
ax_rna = plt.subplot2grid((3, 1), (1, 0), colspan=1)
ax_ratio = plt.subplot2grid((3, 1), (2, 0), colspan=1)


for afd_ratio_i_idx, afd_ratio_i in enumerate(afd_ratio_i_all):

    
    afd_rna_i = afd_rna_i_all[afd_ratio_i_idx]
    afd_dna_i = afd_dna_i_all[afd_ratio_i_idx]

    log_afd_dna_i = numpy.log10(afd_dna_i)
    log_afd_rna_i = numpy.log10(afd_rna_i)
    log_afd_ratio_i = numpy.log10(afd_ratio_i)

    #rescaled_log_afd_dna_i = (log_afd_dna_i - numpy.mean(log_afd_dna_i))/numpy.std(log_afd_dna_i)
    #rescaled_log_afd_rna_i = (log_afd_rna_i - numpy.mean(log_afd_rna_i))/numpy.std(log_afd_rna_i)
    #rescaled_log_afd_ratio_i = (log_afd_ratio_i - numpy.mean(log_afd_ratio_i))/numpy.std(log_afd_ratio_i)

    # rescale by mean
    rescaled_log_afd_dna_i = afd_dna_i/numpy.mean(afd_dna_i)
    rescaled_log_afd_rna_i = afd_rna_i/numpy.mean(afd_rna_i)
    rescaled_log_afd_ratio_i = afd_ratio_i/numpy.mean(afd_ratio_i)

    color = numpy.random.rand(3,)



    ax_dna.scatter(days, rescaled_log_afd_dna_i, s=5, alpha=1, zorder=2, c=color)
    ax_dna.plot(days, rescaled_log_afd_dna_i, lw=1, ls='-', alpha=0.5, c=color, zorder=1)

    ax_rna.scatter(days, rescaled_log_afd_rna_i, s=5, alpha=1, zorder=2, c=color)
    ax_rna.plot(days, rescaled_log_afd_rna_i, lw=1, ls='-', alpha=0.5, c=color, zorder=1)

    ax_ratio.scatter(days, rescaled_log_afd_ratio_i, s=5, alpha=1, zorder=2, c=color)
    ax_ratio.plot(days, rescaled_log_afd_ratio_i, lw=1, ls='-', alpha=0.5, c=color, zorder=1)


ax_dna.set_yscale('log', basey=10)
ax_rna.set_yscale('log', basey=10)
ax_ratio.set_yscale('log', basey=10)


ax_dna.tick_params(axis='both', labelsize=6)
ax_rna.tick_params(axis='both', labelsize=6)
ax_ratio.tick_params(axis='both', labelsize=6)


ax_dna.set_xlabel("Time (days)", fontsize = 9)
ax_rna.set_xlabel("Time (days)", fontsize = 9)
ax_ratio.set_xlabel("Time (days)", fontsize = 9)


#ax_dna.set_ylabel("Rescaled log relative abundance, DNA", fontsize = 7)
#ax_rna.set_ylabel("Rescaled log relative abundance, RNA", fontsize = 7)
#ax_ratio.set_ylabel("Rescaled log relative abundance, RNA/DNA", fontsize = 7)

ax_dna.set_ylabel("Rescaled relative abundance, DNA", fontsize = 7)
ax_rna.set_ylabel("Rescaled relative abundance, RNA", fontsize = 7)
ax_ratio.set_ylabel("Rescaled relative abundance, RNA/DNA", fontsize = 7)





fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%srescaled_rna_dna_ratio_temporal.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

