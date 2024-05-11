import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

numpy.seterr(divide='ignore', invalid='ignore')
min_n_obs = 10



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

# temperature autocorrelation
water_temp_non_nan_idx = ~numpy.isnan(water_temp)
water_temp_non_nan_days = days[water_temp_non_nan_idx]
water_temp_non_nan = water_temp[water_temp_non_nan_idx]

delta_t_water = []
rho_water = []
for i in range(1, len(water_temp_non_nan)-min_n_obs+1):
        
    water_temp_non_nan_t = water_temp_non_nan[i:]

    water_temp_non_nan_delta_t = water_temp_non_nan[:-i]

    delta_t_i = water_temp_non_nan_days[i] - water_temp_non_nan_days[0]

    rho_water_i = numpy.corrcoef(water_temp_non_nan_t, water_temp_non_nan_delta_t)[0,1]

    delta_t_water.append(delta_t_i)
    rho_water.append(rho_water_i)



occupancy_rna = numpy.sum((rel_s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
occupancy_dna = numpy.sum((rel_s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)

# distributio of correlation coefficients for all, keep pairs of OTUs with at least 10 shared timepoints
lower_occupancy_idx = (occupancy_rna==1) & (occupancy_dna==1)
rel_s_by_s_rna_i = rel_s_by_s_rna[lower_occupancy_idx,:]
rel_s_by_s_dna_i = rel_s_by_s_dna[lower_occupancy_idx,:]

mad_rna_i = numpy.mean(rel_s_by_s_rna_i, axis=1)
mad_dna_i = numpy.mean(rel_s_by_s_dna_i, axis=1)

mad_ratio_i = mad_rna_i/mad_dna_i

rescaled_rel_s_by_s_rna_i = (rel_s_by_s_rna_i.T/mad_rna_i).T
rescaled_rel_s_by_s_dna_i = (rel_s_by_s_dna_i.T/mad_dna_i).T

rescaled_rel_s_by_s_ratio_i = rescaled_rel_s_by_s_rna_i/rescaled_rel_s_by_s_dna_i
n_otus = rescaled_rel_s_by_s_ratio_i.shape[0]


rho_all = []
rho_rna_all = []
rho_dna_all = []
delta_t_all = []
n_timepoints = rescaled_rel_s_by_s_ratio_i.shape[1]

for afd_idx, afd in enumerate(rescaled_rel_s_by_s_ratio_i):

    #print(afd)
    afd_dna = rescaled_rel_s_by_s_dna_i[afd_idx,:]
    afd_rna = rescaled_rel_s_by_s_rna_i[afd_idx,:]
    
    #print(numpy.corrcoef(numpy.lib.stride_tricks.sliding_window_view(afd, 5), numpy.tile(numpy.arange(5), (len(afd)-5+1, 1)))[:len(afd)-5+1, -1])
    for i in range(1, n_timepoints-min_n_obs+1):
        
        afd_t = afd[i:]
        afd_dna_t = afd_dna[i:]
        afd_rna_t = afd_rna[i:]

        afd_delta_t = afd[:-i]
        afd_delta_dna_t = afd_dna[:-i]
        afd_delta_rna_t = afd_rna[:-i]

        delta_t_i = days[i] - days[0]

        rho_i = numpy.corrcoef(afd_t, afd_delta_t)[0,1]
        rho_dna_i = numpy.corrcoef(afd_dna_t, afd_delta_dna_t)[0,1]
        rho_rna_i = numpy.corrcoef(afd_rna_t, afd_delta_rna_t)[0,1]

        delta_t_all.append(delta_t_i)
        rho_all.append(rho_i)
        rho_dna_all.append(rho_dna_i)
        rho_rna_all.append(rho_rna_i)



fig = plt.figure(figsize = (12, 4))
fig.subplots_adjust(bottom= 0.15)
ax_ratio = plt.subplot2grid((1, 3), (0, 0), colspan=1)
ax_dna = plt.subplot2grid((1, 3), (0, 1), colspan=1)
ax_rna = plt.subplot2grid((1, 3), (0, 2), colspan=1)

ax_ratio.scatter(delta_t_all, rho_all, s=5, c='k', alpha=0.3)
ax_ratio.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 10)
ax_ratio.set_ylabel("Temporal autocorrelation, " + r'$\rho(\phi_{i} (t), \phi_{i} (t + \Delta t))$', fontsize = 10)
ax_ratio.plot(delta_t_water, rho_water, lw=2, c='dodgerblue', label='Water temperature')
ax_ratio.legend(loc='lower right')

ax_dna.scatter(delta_t_all, rho_dna_all, s=5, c='k', alpha=0.3)
ax_dna.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 10)
ax_dna.set_ylabel("Temporal autocorrelation\nrescaled DNA, " + r'$\rho(d_{i} (t), d_{i} (t + \Delta t))$', fontsize = 10)
ax_dna.plot(delta_t_water, rho_water, lw=2, c='dodgerblue', label='Water temperature')

ax_rna.scatter(delta_t_all, rho_rna_all, s=5, c='k', alpha=0.3)
ax_rna.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 10)
ax_rna.set_ylabel("Temporal autocorrelation\nrescaled RNA, " + r'$\rho(r_{i} (t), r_{i} (t + \Delta t))$', fontsize = 10)
ax_rna.plot(delta_t_water, rho_water, lw=2, c='dodgerblue', label='Water temperature')





fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%sautocorrelation.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


# remove OTUs with all zeros or nans
# identify OTUs where the numbers of observations that (not finite (nan) or (|) equal to zero) is equal to length of timeseries
#to_keep_i_idx = ~(numpy.sum(~numpy.isfinite(rescaled_rel_s_by_s_ratio_i) | (rescaled_rel_s_by_s_ratio_i==0), axis=1) == len(days))

#rescaled_rel_s_by_s_ratio_i = rescaled_rel_s_by_s_ratio_i[to_keep_i_idx,:]
#n_otus_lower = rescaled_rel_s_by_s_ratio_i.shape[0]

