
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


import sine_parameter_utils
import plot_compare_otu1_param_doc





gam_coeff_dict = utils.build_gam_coeff_dict()


doc_coeff_dna = []
doc_coeff_rna = []

for otu_label, otu_dict in gam_coeff_dict.items():
    doc_coeff_dna.append(otu_dict['dna']['doc']['coeff'])
    doc_coeff_rna.append(otu_dict['rna']['doc']['coeff'])


min_ = min(doc_coeff_dna + doc_coeff_rna) - 0.1
max_ = max(doc_coeff_dna + doc_coeff_rna) + 0.1


fig, ax = plt.subplots(figsize=(4.3,4))
ax.scatter(doc_coeff_dna, doc_coeff_rna, c='k', alpha=0.8, s=20, zorder=2, label='One OTU')
ax.plot([min_, max_], [min_, max_], lw=2, c='k', ls=':', zorder=1, label='1:1')

ax.set_xlim([min_, max_])
ax.set_ylim([min_, max_])

ax.set_xlabel("DOC GAM coefficient, DNA", fontsize=11)
ax.set_ylabel("DOC GAM coefficient, RNA", fontsize=12)


ticks = [-1, 0, 1, 2]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(ticks)
ax.set_yticklabels(ticks)
ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)


ax.legend(loc='upper left', fontsize=9)


fig.subplots_adjust(hspace=0.3, wspace=0.25)
fig_name = "%sdna_vs_rna_doc_coeff.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()