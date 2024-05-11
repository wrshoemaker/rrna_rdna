import config
import numpy
import utils
from scipy import stats
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
occupancy_idx = (occupancy_rna==1) & (occupancy_dna==1)
rel_s_by_s_rna_i = rel_s_by_s_rna[occupancy_idx,:]
rel_s_by_s_dna_i = rel_s_by_s_dna[occupancy_idx,:]

mad_rna_i = numpy.mean(rel_s_by_s_rna_i, axis=1)
mad_dna_i = numpy.mean(rel_s_by_s_dna_i, axis=1)

mad_ratio_i = mad_rna_i/mad_dna_i

rescaled_rel_s_by_s_rna_i = (rel_s_by_s_rna_i.T/mad_rna_i).T
rescaled_rel_s_by_s_dna_i = (rel_s_by_s_dna_i.T/mad_dna_i).T

rho_rna = numpy.corrcoef(rescaled_rel_s_by_s_rna_i)
rho_dna = numpy.corrcoef(rescaled_rel_s_by_s_dna_i)

rho_rna_flat = rho_rna[numpy.triu_indices(rho_rna.shape[0], k = 1)]
rho_dna_flat = rho_dna[numpy.triu_indices(rho_dna.shape[0], k = 1)]


#fig, ax = plt.subplots(figsize=(4,4))

fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

ax_hist = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax_scatter = plt.subplot2grid((1, 2), (0, 1), colspan=1)

ax_hist.hist(rho_dna_flat, 20, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['DNA'], label='DNA, occupancy ' + r'$=1$')
ax_hist.hist(rho_rna_flat, 20, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict['RNA'], label='RNA, occupancy ' + r'$=1$')

ax_hist.set_xlabel("Correlation between pairs of rescaled OTUs", fontsize = 9)
ax_hist.set_ylabel("Probability density", fontsize = 10)

ax_hist.axvline(x=0, lw=2, ls=':', c='k')
ax_hist.legend(loc='upper left')


# plot scatter

slope, intercept, r_value, p_value, std_err = stats.linregress(rho_dna_flat, rho_rna_flat)

t_value = (slope - 1)/std_err
p_value = stats.t.sf(numpy.abs(t_value), len(rho_dna_flat)-2)

print(slope, t_value, p_value)



ax_scatter.scatter(rho_dna_flat, rho_rna_flat, alpha=0.4, s=3, c='k', zorder=2)
ax_scatter.plot([-1,1], [-1,1], lw=2, ls=':', c='k', zorder=2)

ax_scatter.set_xlim([-1,1])
ax_scatter.set_ylim([-1,1])

ax_scatter.set_xlabel("Correlation b/w pairs of rescaled OTUs, DNA", fontsize = 9)
ax_scatter.set_ylabel("Correlation b/w pairs of rescaled OTUs, RNA", fontsize = 9)



fig.subplots_adjust(hspace=0.25,wspace=0.3)
fig_name = "%srna_dna_rho.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

