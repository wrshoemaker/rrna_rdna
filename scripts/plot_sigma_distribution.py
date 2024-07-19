import config
import numpy
import utils
from scipy import stats
import matplotlib.pyplot as plt

s_by_s, otu_labels, samples = utils.load_count_data()
# s_by_s.shape = (246, 134265)

metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

n_bins = 30

fig, ax = plt.subplots(figsize=(4,4))

for data_type_i in ['RNA', 'DNA']:

    sample_type_i_idx = (sample_type==data_type_i)
    #sample_type_i = samples[sample_type_i_idx]

    s_by_s_i = s_by_s[:,sample_type_i_idx]

    k, sigma, ids, ids2, meanrelabd = utils.estimate_k_and_sigma(s_by_s_i, min_occupancy=0.2)

    sigma = sigma[ids]
    k = k[ids]

    print(len(sigma))
    # distribution of sigma squared
    sigma_hist_to_plot, sigma_bins_mean_to_plot = utils.get_hist_and_bins(sigma**2, bins=n_bins)

    ax.plot(sigma_bins_mean_to_plot, sigma_hist_to_plot, ls='-', lw=1, c=utils.dna_rna_color_dict[data_type_i], zorder=1)
    ax.scatter(sigma_bins_mean_to_plot, sigma_hist_to_plot, s=30, facecolors=utils.dna_rna_color_dict[data_type_i], edgecolors='k', linewidth=1, alpha=0.9, zorder=2)


ax.set_yscale('log', basey=10)
ax.set_xlabel(r'$\sigma^{2}$', fontsize = 12)
ax.set_ylabel("Probability density, " + r'$P(\sigma^{2})$', fontsize = 9)

ax.tick_params(axis='both', which='minor', labelsize=6)
ax.tick_params(axis='both', which='major', labelsize=6)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssigma_dist.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()
