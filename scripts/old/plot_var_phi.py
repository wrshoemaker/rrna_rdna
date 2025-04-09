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

# distributio of correlation coefficients for all, keep pairs of OTUs with at least 10 shared timepoints
lower_occupancy_idx = (occupancy_rna>=0.1) & (occupancy_dna>=0.1)
rel_s_by_s_rna_i = rel_s_by_s_rna[lower_occupancy_idx,:]
rel_s_by_s_dna_i = rel_s_by_s_dna[lower_occupancy_idx,:]

mad_rna_i = numpy.mean(rel_s_by_s_rna_i, axis=1)
mad_dna_i = numpy.mean(rel_s_by_s_dna_i, axis=1)

mad_ratio_i = mad_rna_i/mad_dna_i

rescaled_rel_s_by_s_rna_i = (rel_s_by_s_rna_i.T/mad_rna_i).T
rescaled_rel_s_by_s_dna_i = (rel_s_by_s_dna_i.T/mad_dna_i).T

rescaled_rel_s_by_s_ratio_i = rescaled_rel_s_by_s_rna_i/rescaled_rel_s_by_s_dna_i
n_otus = rescaled_rel_s_by_s_ratio_i.shape[0]

# remove OTUs with all zeros or nans
# identify OTUs where the numbers of observations that (not finite (nan) or (|) equal to zero) is equal to length of timeseries
to_keep_i_idx = ~(numpy.sum(~numpy.isfinite(rescaled_rel_s_by_s_ratio_i) | (rescaled_rel_s_by_s_ratio_i==0), axis=1) == len(days))

rescaled_rel_s_by_s_ratio_i = rescaled_rel_s_by_s_ratio_i[to_keep_i_idx,:]
n_otus_lower = rescaled_rel_s_by_s_ratio_i.shape[0]

var_ratio_all = []
mean_ratio_all = []
mean_rna_all = []
mean_dna_all = []
for otu_i_idx in range(n_otus_lower):
    
    afd_i = rescaled_rel_s_by_s_ratio_i[otu_i_idx,:]

    idx_to_keep_i = numpy.isfinite(afd_i) 
    #idx_to_keep_i = numpy.isfinite(afd_i) & (afd_i!=0) & (afd_i!=1)
    if sum(idx_to_keep_i) < min_timetpoints_ij:
        continue

    var_i = numpy.var(afd_i[idx_to_keep_i])

    var_ratio_all.append(var_i)
    mean_ratio_all.append(mad_ratio_i[otu_i_idx])
    mean_rna_all.append(mad_rna_i[otu_i_idx])
    mean_dna_all.append(mad_dna_i[otu_i_idx])




fig = plt.figure(figsize = (12, 4))
fig.subplots_adjust(bottom= 0.15)
ax_dna = plt.subplot2grid((1, 3), (0, 0), colspan=1)
ax_rna = plt.subplot2grid((1, 3), (0, 1), colspan=1)
ax_ratio = plt.subplot2grid((1, 3), (0, 2), colspan=1)

# ax_dna
ax_dna.scatter(mean_dna_all, var_ratio_all, s=6, color='k', alpha=0.3, zorder=2)
ax_dna.set_xscale('log', basex=10)
ax_dna.set_yscale('log', basey=10)
ax_dna.set_xlabel("Mean relative abundance, DNA, t", fontsize=10)
ax_dna.set_ylabel('Variance of RNA and DNA ratio', fontsize=10)

# ax_rna
ax_rna.scatter(mean_rna_all, var_ratio_all, s=6, color='k', alpha=0.3, zorder=2)
ax_rna.set_xscale('log', basex=10)
ax_rna.set_yscale('log', basey=10)
ax_rna.set_xlabel("Mean relative abundance, RNA", fontsize=10)
ax_rna.set_ylabel('Variance of RNA and DNA ratio', fontsize=10)

# ax_ratio
ax_ratio.scatter(mean_ratio_all, var_ratio_all, s=6, color='k', alpha=0.3, zorder=2)
ax_ratio.set_xscale('log', basex=10)
ax_ratio.set_yscale('log', basey=10)
ax_ratio.set_xlabel("Ratio of mean RNA and DNA", fontsize=10)
ax_ratio.set_ylabel('Variance of RNA and DNA ratio', fontsize=10)


fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%svar_phi.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

