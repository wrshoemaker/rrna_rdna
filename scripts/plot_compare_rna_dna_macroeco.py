import config
import sys
import argparse
import copy
import numpy
import utils
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy import stats, signal

# numdifftools also installed
import pickle

import simulation_utils


numpy.random.seed(123456789)
n_iter = 1000

s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)

metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

sample_type_rna_idx = (sample_type=='RNA')
sample_type_dna_idx = (sample_type=='DNA')

sample_type_rna = samples[sample_type_rna_idx]

rel_s_by_s_rna = rel_s_by_s[:,sample_type_rna_idx]
rel_s_by_s_dna = rel_s_by_s[:,sample_type_dna_idx]

# non_zero in both datasets

occupancy_rna = numpy.sum(rel_s_by_s_rna>0, axis=1)/rel_s_by_s_rna.shape[1]
occupancy_dna = numpy.sum(rel_s_by_s_dna>0, axis=1)/rel_s_by_s_dna.shape[1]

otu_to_keep_idx = (occupancy_rna>0) & (occupancy_dna>0) 
occupancy_rna = occupancy_rna[otu_to_keep_idx]
occupancy_dna = occupancy_dna[otu_to_keep_idx]
rel_s_by_s_rna = rel_s_by_s_rna[otu_to_keep_idx,:]
rel_s_by_s_dna = rel_s_by_s_dna[otu_to_keep_idx,:]


fig = plt.figure(figsize = (8, 12))
fig.subplots_adjust(bottom= 0.15)


ax_mad_dist = plt.subplot2grid((3, 2), (0, 0), colspan=1)
ax_mad_scatter = plt.subplot2grid((3, 2), (0, 1), colspan=1)

ax_occupancy_dist = plt.subplot2grid((3, 2), (1, 0), colspan=1)
ax_occupancy_scatter = plt.subplot2grid((3, 2), (1, 1), colspan=1)

ax_corr_dist = plt.subplot2grid((3, 2), (2, 0), colspan=1)
ax_corr_scatter = plt.subplot2grid((3, 2), (2, 1), colspan=1)


ax_mad_dist.text(-0.1, 1.07, utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_mad_dist.transAxes)
ax_mad_scatter.text(-0.1, 1.07, utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_mad_scatter.transAxes)

ax_occupancy_dist.text(-0.1, 1.07, utils.sub_plot_labels[2], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_occupancy_dist.transAxes)
ax_occupancy_scatter.text(-0.1, 1.07, utils.sub_plot_labels[3], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_occupancy_scatter.transAxes)

ax_corr_dist.text(-0.1, 1.07, utils.sub_plot_labels[4], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_corr_dist.transAxes)
ax_corr_scatter.text(-0.1, 1.07, utils.sub_plot_labels[5], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_corr_scatter.transAxes)



# plot MAD dist
mad_rna = numpy.mean(rel_s_by_s_rna, axis=1)
mad_dna = numpy.mean(rel_s_by_s_dna, axis=1)

log_mad_rna = numpy.log10(mad_rna)
log_mad_dna = numpy.log10(mad_dna)

rescaled_log_mad_rna = (log_mad_rna - numpy.mean(log_mad_rna))/numpy.std(log_mad_rna)
rescaled_log_mad_dna = (log_mad_dna - numpy.mean(log_mad_dna))/numpy.std(log_mad_dna)

hist_mad_dna, bins_mean_mad_dna = utils.get_hist_and_bins(rescaled_log_mad_dna, bins=100)
hist_mad_rna, bins_mean_mad_rna = utils.get_hist_and_bins(rescaled_log_mad_rna, bins=100)

ax_mad_dist.scatter(bins_mean_mad_dna, hist_mad_dna, s=7, color='dodgerblue', alpha=0.5, lw=1, label='rDNA')
ax_mad_dist.scatter(bins_mean_mad_rna, hist_mad_rna, s=7, color='#FF6347', alpha=0.5, lw=1, label='rRNA')
ax_mad_dist.set_yscale('log', base=10)
ax_mad_dist.xaxis.set_tick_params(labelsize=7)
ax_mad_dist.yaxis.set_tick_params(labelsize=7)
ax_mad_dist.set_xlabel("Rescaled " + r'$\mathrm{log}_{10}$' + " mean relative abundance", fontsize = 10)
ax_mad_dist.set_ylabel("Probability density", fontsize = 10)
ax_mad_dist.legend(loc="upper right")

# plot MAD comparison
ax_mad_scatter.scatter(mad_dna, mad_rna, s=1, color='k', alpha=0.1)
min_mad = min(numpy.concatenate([mad_dna, mad_rna]))
ax_mad_scatter.plot([min_mad,1],[min_mad,1], lw=2, ls=':',c='k', zorder=2, label='1:1')

ax_mad_scatter.set_xlabel("Mean relative abundance, rDNA", fontsize = 10)
ax_mad_scatter.set_ylabel("Mean relative abundance, rRNA", fontsize = 10)
ax_mad_scatter.set_xscale('log', base=10)
ax_mad_scatter.set_yscale('log', base=10)
ax_mad_scatter.xaxis.set_tick_params(labelsize=7)
ax_mad_scatter.yaxis.set_tick_params(labelsize=7)
ax_mad_scatter.set_xlim([min_mad,1])
ax_mad_scatter.set_ylim([min_mad,1])

rho_mad = numpy.corrcoef(log_mad_rna, log_mad_dna)[0,1]
ax_mad_scatter.text(0.2, 0.82, r'$\rho^{2}=$' + str(round(rho_mad**2, 3)), fontsize=10, ha='center', va='center', transform=ax_mad_scatter.transAxes)
ax_mad_scatter.legend(loc="upper left")


# plot occupancy dist
hist_occupancy_dna, bins_mean_occupancy_dna = utils.get_hist_and_bins(occupancy_dna, bins=100)
hist_occupancy_rna, bins_mean_occupancy_rna = utils.get_hist_and_bins(occupancy_rna, bins=100)
ax_occupancy_dist.scatter(bins_mean_occupancy_dna, hist_occupancy_dna, s=7, color='dodgerblue', alpha=0.5, lw=1, label='DNA')
ax_occupancy_dist.scatter(bins_mean_occupancy_rna, hist_occupancy_rna, s=7, color='#FF6347', alpha=0.5, lw=1, label='RNA')
ax_occupancy_dist.set_yscale('log', base=10)
ax_occupancy_dist.xaxis.set_tick_params(labelsize=7)
ax_occupancy_dist.yaxis.set_tick_params(labelsize=7)
ax_occupancy_dist.set_xlabel('Occupancy', fontsize = 10)
ax_occupancy_dist.set_ylabel("Probability density", fontsize = 10)


# plot occupancy scatter
ax_occupancy_scatter.scatter(occupancy_dna, occupancy_rna, s=1, color='k', alpha=0.1)
min_occupancy = min(numpy.concatenate([occupancy_dna, occupancy_rna]))
ax_occupancy_scatter.plot([min_occupancy,1],[min_occupancy,1], lw=2, ls=':',c='k', zorder=2, label='1:1')
ax_occupancy_scatter.set_xlabel("Occupancy, rDNA", fontsize = 10)
ax_occupancy_scatter.set_ylabel("Occupancy, rRNA", fontsize = 10)

ax_occupancy_scatter.set_xscale('log', base=10)
ax_occupancy_scatter.set_yscale('log', base=10)
ax_occupancy_scatter.xaxis.set_tick_params(labelsize=7)
ax_occupancy_scatter.yaxis.set_tick_params(labelsize=7)
ax_occupancy_scatter.set_xlim([min_occupancy,1])
ax_occupancy_scatter.set_ylim([min_occupancy,1])

rho_occupancy = numpy.corrcoef(numpy.log10(occupancy_dna), numpy.log10(occupancy_rna))[0,1]
ax_occupancy_scatter.text(0.2, 0.82, r'$\rho^{2}=$' + str(round(rho_occupancy**2, 3)), fontsize=10, ha='center', va='center', transform=ax_occupancy_scatter.transAxes)




# plot correlation dist
occupancy_one_idx = (occupancy_rna==1) & (occupancy_dna==1)
rel_s_by_s_rna_one = rel_s_by_s_rna[occupancy_one_idx,:]
rel_s_by_s_dna_one = rel_s_by_s_dna[occupancy_one_idx,:]

mad_rna_one = numpy.mean(rel_s_by_s_rna_one, axis=1)
mad_dna_one = numpy.mean(rel_s_by_s_dna_one, axis=1)

rescaled_rel_s_by_s_rna_one = (rel_s_by_s_rna_one.T/mad_rna_one).T
rescaled_rel_s_by_s_dna_one = (rel_s_by_s_dna_one.T/mad_dna_one).T

rho_rna = numpy.corrcoef(rescaled_rel_s_by_s_rna_one)
rho_dna = numpy.corrcoef(rescaled_rel_s_by_s_dna_one)

rho_rna_flat = rho_rna[numpy.triu_indices(rho_rna.shape[0], k = 1)]
rho_dna_flat = rho_dna[numpy.triu_indices(rho_dna.shape[0], k = 1)]

hist_corr_dna, bins_mean_corr_dna = utils.get_hist_and_bins(rho_dna_flat, bins=20)
hist_corr_rna, bins_mean_corr_rna = utils.get_hist_and_bins(rho_rna_flat, bins=20)
ax_corr_dist.scatter(bins_mean_corr_dna, hist_corr_dna, s=18, color='dodgerblue', alpha=0.8, lw=1, label='DNA', zorder=2)
ax_corr_dist.scatter(bins_mean_corr_rna, hist_corr_rna, s=18, color='#FF6347', alpha=0.8, lw=1, label='RNA', zorder=2)
ax_corr_dist.plot(bins_mean_corr_dna, hist_corr_dna, lw=1, color='dodgerblue', alpha=0.8, zorder=1)
ax_corr_dist.plot(bins_mean_corr_rna, hist_corr_rna, lw=1, color='#FF6347', alpha=0.8, zorder=1)
ax_corr_dist.set_yscale('log', base=10)

ax_corr_dist.xaxis.set_tick_params(labelsize=7)
ax_corr_dist.yaxis.set_tick_params(labelsize=7)
ax_corr_dist.set_xlabel("Correlation between ASVs", fontsize = 10)
ax_corr_dist.set_ylabel("Probability density", fontsize = 10)



# plot correlation scatter


ax_corr_scatter.scatter(rho_dna_flat, rho_rna_flat, alpha=0.4, s=3, c='k', zorder=2)
ax_corr_scatter.plot([-1,1], [-1,1], lw=2, ls=':', c='k', zorder=2)
ax_corr_scatter.set_xlabel("Correlation between ASVs, rDNA", fontsize = 10)
ax_corr_scatter.set_ylabel("Correlation between ASVs, rRNA", fontsize = 10)

ax_corr_scatter.xaxis.set_tick_params(labelsize=7)
ax_corr_scatter.yaxis.set_tick_params(labelsize=7)
ax_corr_scatter.set_xlim([-1,1])
ax_corr_scatter.set_ylim([-1,1])

slope, intercept, r_value, p_value, std_err = stats.linregress(rho_dna_flat, rho_rna_flat)

slope_null_all = []
for i in range(n_iter):

    numpy.random.shuffle(rho_dna_flat)
    numpy.random.shuffle(rho_rna_flat)

    slope_null_all.append(stats.linregress(rho_dna_flat, rho_rna_flat)[0])


t_value = (slope - 1)/std_err
p_value = stats.t.sf(numpy.abs(t_value), len(rho_dna_flat)-2)




#ax_corr_scatter.text(0.2, 0.62, r'$\rho^{2}=$' + str(round(r_value**2, 3)), fontsize=10, ha='center', va='center', transform=ax_corr_scatter.transAxes)
ax_corr_scatter.text(0.2, 0.82, 'Slope = ' + str(round(slope, 3)), fontsize=10, ha='center', va='center', transform=ax_corr_scatter.transAxes)
#ax_corr_scatter.text(0.2, 0.72, r'$ P \, \nleq \, 0.05$', fontsize=10, ha='center', va='center', transform=ax_corr_scatter.transAxes)
ax_corr_scatter.text(0.2, 0.72, r'$ P = $' + str(round(p_value, 3)), fontsize=10, ha='center', va='center', transform=ax_corr_scatter.transAxes)




#ax_scatter.set_ylabel("Correlation b/w pairs of rescaled OTUs, RNA", fontsize = 9)


fig.subplots_adjust(hspace=0.2, wspace=0.3)
fig_name = "%scompare_rna_dna_macroeco.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
