import config
import sys
import random
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

import sine_parameter_utils


fig, ax = plt.subplots(figsize=(5,4))



ax.text(0.1, 0.8, r'$\tau_{\mathrm{photo}}^{\mathrm{env}}$', fontsize=24, transform=ax.transAxes)

ax.text(0.1, 0.6, r'$\tau_{\mathrm{i}}^{\mathrm{env}}$', fontsize=24, transform=ax.transAxes)


ax.text(0.1, 0.4, r'$\Delta \psi_{i} \equiv \psi_{i}^{\mathrm{RNA}} -\psi_{i}^{\mathrm{DNA}}$', fontsize=24, transform=ax.transAxes)

ax.text(0.1, 0.1, r'$A_{i}$', fontsize=24, transform=ax.transAxes)


fig.subplots_adjust(hspace=0.4, wspace=0.35)
fig_name = "%stext_for_figs.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
