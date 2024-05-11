import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

numpy.seterr(divide='ignore', invalid='ignore')
min_timetpoints_ij = 10



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

rho_rescaled_ratio = numpy.corrcoef(rescaled_rel_s_by_s_ratio_subset)
rho_rescaled_ratio_flat = rho_rescaled_ratio[numpy.triu_indices(rho_rescaled_ratio.shape[0], k = 1)]

fig, ax = plt.subplots(figsize=(4,4))


ax.hist(rho_rescaled_ratio_flat, 20, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['DNA'], label='Occupancy ' + r'$=1$')

ax.set_xlabel("Correlation, " + r'$\rho(\phi_{i} (t), \phi_{j} (t))$', fontsize = 10)
ax.set_ylabel("Probability density", fontsize = 10)

ax.axvline(x=0, lw=2, ls=':', c='k')


#ax.legend(loc="upper left")

# distributio of correlation coefficients for all, keep pairs of OTUs with at least 10 shared timepoints
lower_occupancy_idx = (occupancy_rna>=0.1) & (occupancy_dna>=0.1)
rel_s_by_s_rna_i = rel_s_by_s_rna[lower_occupancy_idx,:]
rel_s_by_s_dna_i = rel_s_by_s_dna[lower_occupancy_idx,:]

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

rho_lower_all = []
n_otus_lower = rescaled_rel_s_by_s_ratio_i.shape[0]
for otu_i_idx in range(n_otus_lower):
    
    afd_i = rescaled_rel_s_by_s_ratio_i[otu_i_idx,:]

    idx_to_keep_i = numpy.isfinite(afd_i) & (afd_i!=0) & (afd_i!=1)
    if sum(idx_to_keep_i) < min_timetpoints_ij:
        continue

    print(otu_i_idx, n_otus_lower)
    
    for otu_j_idx in range(otu_i_idx):

        afd_j = rescaled_rel_s_by_s_ratio_i[otu_j_idx,:]
        #idx_to_keep_ij = (numpy.isfinite(afd_i) & numpy.isfinite(afd_j)) & (afd_i!=0) & (afd_j!=0) & (afd_i!=1) & (afd_j!=1)  
        idx_to_keep_ij = (numpy.isfinite(afd_i) & numpy.isfinite(afd_j))

        if sum(idx_to_keep_ij) < min_timetpoints_ij:
            continue

        rho_lower_all.append(numpy.corrcoef(afd_i[idx_to_keep_ij], afd_j[idx_to_keep_ij ])[0,1]  )


ax.hist(rho_lower_all, 20, histtype='step',density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['RNA'], label='Occupancy ' + r'$\geq 0.1$')

ax.legend(loc="upper left")

fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%sphi_rho.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
