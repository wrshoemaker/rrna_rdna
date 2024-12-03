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


cv_afd_rna_i_all = []
cv_afd_dna_i_all = []
cv_log_ratio_rna_i_all = []
cv_log_ratio_dna_i_all = []
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
    #print(afd_ratio_i)

    cv_afd_rna_i = numpy.std(afd_rna_i)/numpy.absolute(numpy.mean(afd_rna_i))
    cv_afd_dna_i = numpy.std(afd_dna_i)/numpy.absolute(numpy.mean(afd_dna_i))
    
    cv_afd_rna_i_all.append(cv_afd_rna_i)
    cv_afd_dna_i_all.append(cv_afd_dna_i)

    # log-ratio
    log_ratio_rna_i = numpy.log10(afd_rna_i[1:]/afd_rna_i[:-1])
    log_ratio_dna_i = numpy.log10(afd_dna_i[1:]/afd_dna_i[:-1])

    cv_log_ratio_rna_i = numpy.std(log_ratio_rna_i)/numpy.absolute(numpy.mean(log_ratio_rna_i))
    cv_log_ratio_dna_i = numpy.std(log_ratio_dna_i)/numpy.absolute(numpy.mean(log_ratio_dna_i))

    cv_log_ratio_rna_i_all.append(cv_log_ratio_rna_i)
    cv_log_ratio_dna_i_all.append(cv_log_ratio_dna_i)


# CV of AFD
fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(cv_afd_rna_i_all, cv_afd_dna_i_all, c='k', s=70, alpha=0.9, edgecolors='none', zorder=2)

data_merged = numpy.concatenate((cv_afd_rna_i_all, cv_afd_dna_i_all), axis=None)
min_data, max_data = min(data_merged), max(data_merged)

ax.plot([min_data, max_data], [min_data, max_data], ls=':', lw='2', c='k', zorder=1)
#ax.scatter(distance_all, rho_all, zorder=2, alpha=0.3)

ax.set_xlim([min_data, max_data])
ax.set_ylim([min_data, max_data])

ax.set_xlabel("CV of AFD, DNA", fontsize = 11)
ax.set_ylabel("CV of AFD, RNA", fontsize = 11)
ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)

fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%scv_afd.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



# CV of log ratio
fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(cv_log_ratio_dna_i_all, cv_log_ratio_rna_i_all, c='k', s=70, alpha=0.9, edgecolors='none', zorder=2)

data_merged = numpy.concatenate((cv_log_ratio_dna_i_all, cv_log_ratio_rna_i_all), axis=None)
min_data, max_data = min(data_merged), max(data_merged)

ax.plot([min_data, max_data], [min_data, max_data], ls=':', lw='2', c='k', zorder=1)
#ax.scatter(distance_all, rho_all, zorder=2, alpha=0.3)

ax.set_xlim([min_data, max_data])
ax.set_ylim([min_data, max_data])

ax.set_xlabel("CV of log-ratio, DNA", fontsize = 11)
ax.set_ylabel("CV of log-ratio, RNA", fontsize = 11)
ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)

fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%scv_log_ratio.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



# time vs. ratio 
# temp vs. ratio

fig = plt.figure(figsize = (10, 8))
fig.subplots_adjust(bottom= 0.15)

ax_time = plt.subplot2grid((4, 1), (0, 0), colspan=1)
ax_temp = plt.subplot2grid((4, 1), (1, 0), colspan=1)
ax_temp_rna = plt.subplot2grid((4, 1), (2, 0), colspan=1)
ax_temp_dna = plt.subplot2grid((4, 1), (3, 0), colspan=1)

for afd_ratio_i_idx, afd_ratio_i in enumerate(afd_ratio_i_all):


    afd_ratio_i_temp = afd_ratio_i[water_temp_non_nan_idx]

    
    
    afd_rna_i_temp = afd_rna_i_all[afd_ratio_i_idx]
    afd_dna_i_temp = afd_dna_i_all[afd_ratio_i_idx]

    afd_rna_i_temp = afd_rna_i_temp[water_temp_non_nan_idx]
    afd_dna_i_temp = afd_dna_i_temp[water_temp_non_nan_idx]

    color = numpy.random.rand(3,)
    #color = color.reshape(1,-1)
    #color = cm.nipy_spectral(float(i) / n_clusters).reshape(1,-1)

    #color = numpy.append(color, 1)
    #print(color)
    #color = numpy.array([color])
    #color = color.reshape(-1,4)

    ax_time.scatter(days, afd_ratio_i, s=5, alpha=1, zorder=2, c=color)
    ax_time.plot(days, afd_ratio_i, lw=1, ls='-', alpha=0.5, c=color, zorder=1)


    water_temp_non_nan_sort_idx = numpy.argsort(water_temp_non_nan)

    water_temp_non_nan = water_temp_non_nan[water_temp_non_nan_sort_idx]
    afd_ratio_i_temp = afd_ratio_i_temp[water_temp_non_nan_sort_idx]
    afd_rna_i_temp = afd_rna_i_temp[water_temp_non_nan_sort_idx]
    afd_dna_i_temp = afd_dna_i_temp[water_temp_non_nan_sort_idx]

    ax_temp.scatter(water_temp_non_nan, afd_ratio_i_temp, s=5, alpha=0.8, c=color,zorder=2)
    ax_temp_rna.scatter(water_temp_non_nan, afd_rna_i_temp, s=5, alpha=0.8, c=color)
    ax_temp_dna.scatter(water_temp_non_nan, afd_dna_i_temp, s=5, alpha=0.8, c=color)

    ax_temp.plot(water_temp_non_nan, afd_ratio_i_temp, lw=1, ls='-', alpha=0.5, c=color, zorder=1)
    ax_temp_rna.plot(water_temp_non_nan, afd_rna_i_temp, lw=1, ls='-', alpha=0.5, c=color, zorder=1)
    ax_temp_dna.plot(water_temp_non_nan, afd_dna_i_temp, lw=1, ls='-', alpha=0.5, c=color, zorder=1)



ax_time.tick_params(axis='both', labelsize=6)
ax_temp.tick_params(axis='both', labelsize=6)
ax_temp_rna.tick_params(axis='both', labelsize=6)
ax_temp_dna.tick_params(axis='both', labelsize=6)


ax_time.set_yscale('log', basey=10)
ax_temp.set_yscale('log', basey=10)
ax_temp_rna.set_yscale('log', basey=10)
ax_temp_dna.set_yscale('log', basey=10)

ax_time.set_xlabel("Time (days)", fontsize = 9)
ax_temp.set_xlabel("Temperature (C)", fontsize = 9)
ax_temp_rna.set_xlabel("Temperature (C)", fontsize = 9)
ax_temp_dna.set_xlabel("Temperature (C)", fontsize = 9)

ax_time.set_ylabel("RNA/DNA relative abundance", fontsize = 7)
ax_temp.set_ylabel("RNA/DNA relative abundance", fontsize = 7)
ax_temp_rna.set_ylabel("RNA relative abundance", fontsize = 7)
ax_temp_dna.set_ylabel("DNA relative abundance", fontsize = 7)



fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%srna_dna_ratio_temporal.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


#print(len(occupancy))