import config
import utils
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy import stats


metadata_dict = utils.build_metadata_dict()

samples_dna, days_dna, mcn_dna = utils.calculate_mean_copy_number('DNA')
samples_rna, days_rna, mcn_rna = utils.calculate_mean_copy_number('RNA')


temp_dna = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples_dna])
temp_rna = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples_rna])

dna_filter_idx = ~numpy.isnan(temp_dna)
rna_filter_idx = ~numpy.isnan(temp_rna)

temp_dna_filter = temp_dna[dna_filter_idx]
temp_rna_filter = temp_rna[rna_filter_idx]

mcn_dna_filter = mcn_dna[dna_filter_idx]
mcn_rna_filter = mcn_rna[rna_filter_idx]

temp_dna_filter_sort_idx = numpy.argsort(temp_dna_filter)
temp_rna_filter_sort_idx = numpy.argsort(temp_rna_filter)


temp_dna_filter_sort = temp_dna_filter[temp_dna_filter_sort_idx]
temp_rna_filter_sort = temp_rna_filter[temp_rna_filter_sort_idx]

mcn_dna_filter_sort = mcn_dna_filter[temp_dna_filter_sort_idx]
mcn_rna_filter_sort = mcn_rna_filter[temp_rna_filter_sort_idx]




# days vs. MCN

fig = plt.figure(figsize = (10, 6))
fig.subplots_adjust(bottom= 0.15)

ax_time = plt.subplot2grid((2, 1), (0, 0), colspan=1)
ax_temp = plt.subplot2grid((2, 1), (1, 0), colspan=1)

# CV of log ratio
ax_time.plot(days_dna, mcn_dna, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
ax_time.scatter(days_dna, mcn_dna, s=5, alpha=0.8, c='k', zorder=1, label='DNA')

ax_time.plot(days_rna, mcn_rna, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
ax_time.scatter(days_rna, mcn_rna, s=20, alpha=0.8, zorder=2, facecolors='white', edgecolors='k', label='RNA')

#ax.ax_time(distance_all, rho_all, zorder=2, alpha=0.3)
ax_time.set_xlabel("Time (days)", fontsize = 9)
ax_time.set_ylabel("Mean rRNA operon copy number", fontsize = 7)
ax_time.legend(loc="lower right")



ax_temp.plot(temp_dna_filter_sort, mcn_dna_filter_sort, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
ax_temp.scatter(temp_rna_filter_sort, mcn_dna_filter_sort, s=5, alpha=0.8, c='k', zorder=1, label='DNA')

ax_temp.plot(temp_dna_filter_sort, mcn_rna_filter_sort, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
ax_temp.scatter(temp_rna_filter_sort, mcn_rna_filter_sort, s=20, alpha=0.8, zorder=2, facecolors='white', edgecolors='k', label='RNA')

#ax.ax_time(distance_all, rho_all, zorder=2, alpha=0.3)
ax_temp.set_xlabel("Temperature (C)", fontsize = 9)
ax_temp.set_ylabel("Mean rRNA operon copy number", fontsize = 7)


fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%stemp_vs_mcn.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()




#rel_s_by_s_copy_number = ((rel_s_by_s_coarse_to_keep.T * b).T)

#print(rel_s_by_s_copy_number)
#print(rel_s_by_s_copy_number.shape)

#print(numpy.mean(numpy.log10(mad_included)), numpy.mean(numpy.log10(mad_excluded)))
#print(mad[coarse_labels_to_keep_idx])


#print(mad[~coarse_labels_to_keep_idx])
#print(coarse_labels[coarse_labels_to_keep_idx])


#coarse_labels_excluded = numpy.delete(coarse_labels, coarse_labels_to_keep_idx)


#print(coarse_labels_excluded[:10])
#print(mad_excluded)




#print(len(intersection) / len(coarse_labels))



#print(coarse_labels_set - intersection)

#print(coarse_labels)


#print(s_by_s_coarse.shape)