import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy import stats


numpy.seterr(divide='ignore', invalid='ignore')
min_n_obs = 10



s_by_s, otu_labels, samples = utils.load_count_data()

# s_by_s.shape = (246, 134265)

s_by_s_dna, s_by_s_rna = utils.subset_s_by_s_occupancy(s_by_s, samples, min_occupancy=0.1)

rel_s_by_s_dna = s_by_s_dna/numpy.sum(s_by_s_dna, axis=0)
rel_s_by_s_rna = s_by_s_rna/numpy.sum(s_by_s_rna, axis=0)

mad_dna = numpy.mean(rel_s_by_s_dna, axis=1)
mad_rna = numpy.mean(rel_s_by_s_rna, axis=1)

# mean over time points of the absolute value of the difference between successive timepoints
mean_delta_dna = numpy.mean(numpy.absolute(rel_s_by_s_rna[:,1:] - rel_s_by_s_rna[:,:-1]), axis=1)
mean_delta_rna = numpy.mean(numpy.absolute(rel_s_by_s_dna[:,1:] - rel_s_by_s_dna[:,:-1]), axis=1)



slope_dna, intercept_dna, r_value_dna, p_value_dna, std_err_dna = stats.linregress(numpy.log10(mad_dna), numpy.log10(mean_delta_dna))
slope_rna, intercept_rna, r_value_rna, p_value_rna, std_err_rna = stats.linregress(numpy.log10(mad_rna), numpy.log10(mean_delta_rna))




# colors
#colormap_dna = utils.make_colormap('DNA', len(mad_dna))
#colors_dna = [colormap_dna[k] for k in range(len(mad_dna))]



fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)
ax_dna = plt.subplot2grid((1, 2), (0, 0), colspan=1)
ax_rna = plt.subplot2grid((1, 2), (0, 1), colspan=1)



ax_dna.scatter(mad_dna, mean_delta_dna, c=utils.dna_rna_color_dict['DNA'], s=4, alpha=0.3)
ax_rna.scatter(mad_rna, mean_delta_rna, c=utils.dna_rna_color_dict['RNA'], s=4, alpha=0.3)


#ax_dna.scatter(mad_dna, mean_delta_dna, alpha=0.9, color=colors_dna, edgecolors='k', s=8, linewidths=0.5)

#ax_dna.scatter(mad_dna, mean_delta_dna, alpha=0.9, color=colors_dna, edgecolors='k', s=8)




# plot regressions
x_log_range_dna =  numpy.linspace(min(numpy.log10(mad_dna)), max(numpy.log10(mad_dna)), 10000)
y_fit_range_dna = slope_dna*x_log_range_dna + intercept_dna

x_log_range_rna =  numpy.linspace(min(numpy.log10(mad_rna)), max(numpy.log10(mad_rna)), 10000)
y_fit_range_rna = slope_rna*x_log_range_rna + intercept_rna

ax_dna.plot(10**x_log_range_dna, 10**y_fit_range_dna, lw=2, ls='--', c='k', zorder=2)
ax_rna.plot(10**x_log_range_rna, 10**y_fit_range_rna, lw=2, ls='--', c='k', zorder=2)

#ax.scatter(mean_trajectory_ratio_i_subset, mean_copy_number_i, s=5, alpha=0.8, c=utils.dna_rna_color_dict[type_i], zorder=1, label=type_i)
ax_dna.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope_dna, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_dna.transAxes)
ax_rna.text(0.2,0.9, r'$y \sim x^{{{}}}$'.format(str( round(slope_rna, 3) )), fontsize=11, color='k', ha='center', va='center', transform=ax_rna.transAxes)




ax_dna.set_xscale('log', basex=10)
ax_dna.set_yscale('log', basey=10)
ax_rna.set_xscale('log', basex=10)
ax_rna.set_yscale('log', basey=10)

ax_dna.set_title('DNA', fontsize=12)
ax_rna.set_title('RNA', fontsize=12)
    

ax_dna.set_xlabel("Mean relative abundance", fontsize=10)
ax_rna.set_xlabel("Mean relative abundance", fontsize=10)

y_label = "Mean absolute difference between " + r'$x_{i}$' + '\nat consecutive timepoints, ' + r'$\left< \left| x_{i}(t + \delta t) - x_{i}(t)  \right| \right>$'
ax_dna.set_ylabel(y_label, fontsize=10)
ax_rna.set_ylabel(y_label, fontsize=10)


fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%smean_vs_time_difference.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


