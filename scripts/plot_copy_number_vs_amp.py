import config
import numpy
import pickle
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.gridspec as gridspec

import sine_parameter_utils

from scipy import stats

taxonomic_level = 'genus'
taxonomy_dict = utils.build_taxonomy_dict()
rrna_copy_dict = utils.make_rrna_copy_dict()
rrna_copy_taxa = numpy.asarray(list(rrna_copy_dict.keys()))

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))

#otu_labels = param_dict[otu_labels'']


genus_param = [taxonomy_dict[k][taxonomic_level] for k in param_dict['otu_labels']]

#print(param_dict['otu_labels'])
#print([taxonomy_dict[k]['class'] for k in param_dict['otu_labels']])

to_keep_idx = []
rrna_copy_number = []
for g_idx, g in enumerate(genus_param):

    if g in rrna_copy_dict:
        to_keep_idx.append(g_idx)
        rrna_copy_number.append(rrna_copy_dict[g])

to_keep_idx = numpy.asarray(to_keep_idx)


fig = plt.figure(figsize = (8.5, 12)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=3, ncols=2)


param_label_dict = {'amp': "amplitude, " + r'$A_{i}$', 'freq':  "oscillation timescale, " + r'$\tau_{i}^{\mathrm{env}}$', 'phase': "phase, " + r'$\psi_{i}$'}

for param_idx, param in enumerate(['amp', 'freq', 'phase']):

    for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

        param_mle = numpy.asarray(param_dict['%s_mle'%param][data_type])[to_keep_idx]

        if param == 'freq':
            param_mle = 2*numpy.pi/param_mle

        ax = fig.add_subplot(gs[param_idx, data_type_idx])

        slope, intercept, r_value, p_value, std_err = stats.linregress(rrna_copy_number, param_mle)
        #x_range_ =  numpy.linspace(min(rrna_copy_number), max(rrna_copy_number), 10000)
        #y_fit_range = slope*x_range_ + intercept
        #ax.plot(x_range_, y_fit_range, ls='--', lw=2.5, c=utils.dna_rna_color_dict[data_type])
        ax.text(0.26, 0.78, utils.get_p_value_latex_label_dict(p_value), fontsize=9, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.26, 0.87, 'Slope = ' + str(round(slope, 3)), fontsize=9, ha='center', va='center', transform=ax.transAxes)

        ax.scatter(rrna_copy_number, param_mle, s=25, alpha=1, color=utils.dna_rna_color_dict[data_type], zorder=2)

        ax.set_xlabel("Mean genus-level rRNA operon copy number", fontsize=10)
        ax.set_ylabel("Inferred " + data_type + ' ' + param_label_dict[param], fontsize=10)

        if param == 'phase':

            phase_ticks = [0, 0.5*numpy.pi, numpy.pi, 1.5*numpy.pi, 2*numpy.pi]
            phase_tick_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
            ax.set_xticks(phase_ticks)
            ax.set_xticklabels(phase_tick_labels)
        
        ax.xaxis.set_tick_params(labelsize=7)
        ax.yaxis.set_tick_params(labelsize=7)



fig.subplots_adjust(hspace=0.4, wspace=0.3)
fig_name = "%scopy_number_vs_amp.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


#print(param_dict['amp_mle'].keys())