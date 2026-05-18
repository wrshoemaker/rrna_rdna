import config
import sys
import pickle
import copy
import numpy
import utils
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sine_parameter_utils

import matplotlib.pyplot as plt
from matplotlib import cm, colors

metadata_dict = utils.build_metadata_dict()
s_by_s, otu_labels, samples = utils.load_count_data()
samples = numpy.asarray(samples)
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])


fig = plt.figure(figsize = (8.5, 4))
fig.subplots_adjust(bottom= 0.15)

for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

    sample_type_rna_idx = (sample_type==data_type)
    s_by_s_rna = s_by_s[:,sample_type_rna_idx]

    occupancy_rna = numpy.sum((s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
    occupancy_range = numpy.linspace(0.8, 1, num=100, endpoint=True)

    slope_all = []
    std_err_all = []
    for o in occupancy_range:

        o_idx = occupancy_rna >= o
        s_by_s_rna_i = s_by_s_rna[o_idx, :]
        all_present_idx = numpy.all(s_by_s_rna_i > 0, axis=0)
        s_by_s_rna_o = s_by_s_rna[:,all_present_idx]
        s_by_s_rna_clr_o, occupancy_clr_idx = utils.clr_transform_sim_subset(s_by_s_rna_o, min_occupancy=1)

        mean_clr_o = numpy.mean(s_by_s_rna_clr_o, axis=1)
        var_clr_o = numpy.var(s_by_s_rna_clr_o, axis=1)
        std_clr_o = numpy.std(s_by_s_rna_clr_o, axis=1)

        std_clr_rms_o = std_clr_o / numpy.mean(numpy.absolute(s_by_s_rna_clr_o), axis=1)

        #slope, intercept, r_value, p_value, std_err = stats.linregress(mean_clr_o, std_clr_o)
        slope_all.append(numpy.mean(std_clr_rms_o))
        std_err_all.append(stats.sem(std_clr_rms_o))



    slope_all = numpy.asarray(slope_all)
    #std_err_all = numpy.asarray(std_err_all)


    ax = plt.subplot2grid((1, 2), (0, data_type_idx))
    ax.plot(occupancy_range, slope_all, c=utils.dna_rna_color_dict[data_type], lw=4, ls='-', zorder=2, label='Mean over ASVs')
    ax.fill_between(occupancy_range, slope_all - std_err_all, slope_all + std_err_all, alpha=0.25, color=utils.dna_rna_color_dict[data_type], label='Std. err.')
    ax.set_xlabel('Minimum ASV occupancy', fontsize=12)
    ax.set_ylabel('Std dev. normalized by mean absolute\nCLR-transformed abund., ' r'$\frac{\sigma_{\hat{c}_{i}}}{\overline{|\hat{c}_{i}|}}$', fontsize=12)
    ax.set_title(utils.rescaled_label_clr_dict[data_type], fontsize=16, color=utils.dna_rna_color_dict[data_type], fontweight='bold')
    #ax.axhline(y=0, ls=':', lw=3, c='k', zorder=3)

    ax.set_xlim([0.8, 1])
    ax.set_ylim([0, 1.1])

    #if data_type_idx == 0:
    ax.legend(loc='lower right')


fig.subplots_adjust(hspace=0.35, wspace=0.45)
fig_name = "%smean_vs_var_by_occupancy.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
