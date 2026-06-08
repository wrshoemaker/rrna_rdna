import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy import stats, signal
# numdifftools also installed
import pickle
import sine_parameter_utils



#def time_delay_small_tan():


intercept = -1.7500593949990497
g_range = numpy.linspace(1, 4, num=1000, endpoint=True)
alpha = 0.8


fig, ax = plt.subplots(figsize=(4.5,4))

ax.plot(g_range, intercept + 1*numpy.log(g_range), lw=2, ls='-', c='#CC4A35', label='Superlinear, ' + r'$\alpha > 0$')
ax.plot(g_range, intercept + 0*numpy.log(g_range), lw=2, ls='-', c='#FF6347', label='Linear, ' + r'$\alpha = 0$')
ax.plot(g_range, intercept + -1*numpy.log(g_range), lw=2, ls='-', c='#FF8A75', label='Sublinear, ' + r'$\alpha < 0$')



ax.legend(loc='upper left', fontsize=9, title='Gene dosage effect', framealpha=1)


fig.subplots_adjust(hspace=0.35, wspace=0.25)
fig_name = "%scopy_num_vs_ratio_theory.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()





# second plot


def analytic_time_lag(alpha_r, sigma_r, gamma, sigma_cell, lambda_n):
    numerator   = alpha_r * sigma_r**2 - gamma * sigma_cell**2
    denominator = gamma**2 * sigma_cell**2 + alpha_r * lambda_n * sigma_r**2
    return numerator / denominator




fig, ax = plt.subplots(figsize=(4.5,4))

gamma    = 1.0
lambda_n = 1.0
sigma_cell = 1.0
noise_ratio = numpy.linspace(0.01, 3.0, 1000)
sigma_r = noise_ratio * sigma_cell



#ls_all = [':', '--', '-']
#label_all = ['Low', 'Equal', 'High']
lag_time_all = []
for alpha_R_idx, alpha_R in enumerate([1/4, 1, 4]):
    lag_time = analytic_time_lag(alpha_R, sigma_r, gamma, sigma_cell, lambda_n)
    lag_time_all.append(lag_time)
    #ax.plot(lag_time, noise_ratio, ls=ls_all[alpha_R_idx], label=label_all[alpha_R_idx], c='k')



#ax.plot(noise_ratio, noise_ratio, ls=ls_all[alpha_R_idx], label=label_all[alpha_R_idx], c='k')


lag_time_plot = analytic_time_lag(1, sigma_r, gamma, sigma_cell, lambda_n)
ax.plot(noise_ratio, lag_time_plot, ls='-', lw=2, c='k')



lag_time_all = numpy.unique(numpy.concatenate(lag_time_all))

#ax.legend(loc='upper left', fontsize=9, title='Strength of RNA-growth coupling dosage effect')

#ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncols=3, borderaxespad=0, fontsize=9, title='Strength of rRNA-growth coupling vs. degredation')


#lag_time_range = numpy.linspace(min(lag_time_all), max(lag_time_all), 1000)

x_min = 0
x_max = 3

y_min = min(lag_time_plot)
y_max = max(lag_time_plot)

ax.set_xlim([0,3])
ax.set_ylim([y_min,y_max])

ax.set_xticklabels([])
ax.set_yticklabels([])

#ax.fill_betweenx(noise_ratio, y_min, y_max, where=lag_time_plot >= 0, facecolor=utils.dna_rna_color_dict['RNA'], alpha=0.5, zorder=1)
#ax.fill_betweenx(noise_ratio, y_min, y_max, where=lag_time_plot <= 0, facecolor=utils.dna_rna_color_dict['DNA'], alpha=0.5, zorder=1)


ax.axhspan(y_min, 0, facecolor=utils.dna_rna_color_dict['DNA'], alpha=0.5, zorder=1)
ax.axhspan(0, y_max, facecolor=utils.dna_rna_color_dict['RNA'], alpha=0.5, zorder=1)

#delta_x = max(noise_ratio) - min(noise_ratio)
#x_max = max(noise_ratio)
#x_min = min(noise_ratio)

#ax.fill_between(x_max, lag_time_range, where=lag_time_range >= 0, facecolor= utils.dna_rna_color_dict['RNA'], alpha=0.5, zorder=1)
#ax.fill_between(x_max, lag_time_range, where=lag_time_range <= 0, facecolor= utils.dna_rna_color_dict['DNA'], alpha=0.5, zorder=1)

#ax.text(0.25, 0.9, 'rDNA ' + r'$\rightarrow$' + ' rRNA', fontsize=13, ha='center', va='center', transform=ax.transAxes)
#ax.text(0.76, 0.1, 'rRNA ' + r'$\rightarrow$' + ' rDNA', fontsize=13, ha='center', va='center', transform=ax.transAxes)


#ax.set_xlim([min(lag_time_all), max(lag_time_all)])
#ax.set_ylim([x_min, x_max])

#trength of rRNA-to-growth coupling relative to rRNA turnover rate


fig.subplots_adjust(hspace=0.35, wspace=0.25)
fig_name = "%snoise_vs_phase_delay.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
