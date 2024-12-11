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


def build_gam_coeff_dict():

    gam_coeff_dict = {}

    gam_env_analysis_path = '%sgam_env_analysis.csv' % config.data_directory

    gam_env_analysis_file = open(gam_env_analysis_path, 'r')
    header = gam_env_analysis_file.readline()
    env_variables = header.strip().split(',')[2:]

    for line in gam_env_analysis_file:

        line = line.strip().split(',')
        otu = line[0].split('_', 1)[0]
        data_type = line[0].split('_', 1)[1]

        if otu not in gam_coeff_dict:
            gam_coeff_dict[otu] = {}

            for d in ['dna', 'rna', 'rna_dna']:
                gam_coeff_dict[otu][d] = {}

                for e in env_variables:
                    gam_coeff_dict[otu][d][e] = {}

        p_value_or_coeff = line[1]

        for e_idx, e in enumerate(env_variables):
            gam_coeff_dict[otu][data_type][e][p_value_or_coeff] = float(line[e_idx+2])

    gam_env_analysis_file.close()


    return gam_coeff_dict
    


param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
gam_coeff_dict = build_gam_coeff_dict()

sine_param_all = []
coeff_all = []
pvalue_all = []

sine_param_to_plot = 'freq_mle'
env_variable_to_plot = 'doc'
data_type_to_plot = 'RNA'
for otu_label_idx, otu_label in enumerate(param_dict['otu_labels']):

    if otu_label_idx == 0:
        continue

    param_otu = gam_coeff_dict[otu_label][data_type_to_plot.lower()][env_variable_to_plot]['coeff']

    sine_param_all.append(param_dict[sine_param_to_plot][data_type_to_plot][otu_label_idx])
    coeff_all.append(param_otu)
    pvalue_all.append(gam_coeff_dict[otu_label][data_type_to_plot.lower()][env_variable_to_plot]['p_value'])


focal_otu_sine_param = param_dict[sine_param_to_plot][data_type_to_plot][0]
focal_otu_coeff = gam_coeff_dict[param_dict['otu_labels'][0]][data_type_to_plot.lower()][env_variable_to_plot]['coeff']


sine_param_all = numpy.asarray(sine_param_all)
coeff_all = numpy.asarray(coeff_all)
pvalue_all = numpy.asarray(pvalue_all)

if 'freq' in sine_param_to_plot:

    sine_param_all = 2*numpy.pi/sine_param_all
    focal_otu_sine_param = 2*numpy.pi/focal_otu_sine_param




def bhp(p_values):
	rank = stats.rankdata(p_values,method='max')
	nbp = len(p_values)
	return [min(p_values[i]*nbp/rank[i],1) for i in range(nbp)]


from statsmodels.stats.multitest import fdrcorrection
#from statsmodels.stats.multitest import multipletests

pvalue_all = fdrcorrection(pvalue_all, alpha=0.05, method='indep', is_sorted=False)[1]




fig, ax = plt.subplots(figsize=(4,4))

ax.scatter(focal_otu_sine_param, focal_otu_coeff, alpha=1, s=30, color='#FF6347', label='OTU 1 (phototroph)')

pvalue_significant_idx = (pvalue_all <= 0.05)

for sig_bool_ in [True, False]:

    # skip if there are no significant slopes...
    if sum(pvalue_significant_idx==sig_bool_) == 0:
        continue

    if sig_bool_ == True:
        sig_bool_label = 'significant'
        color = '#87CEEB'
    else:
        sig_bool_label = 'nonsignificant'
        color = 'k'


    ax.scatter(sine_param_all[pvalue_significant_idx==sig_bool_], coeff_all[pvalue_significant_idx==sig_bool_], alpha=0.8, s=20, color=color, label='Other OTUs (%s)' % sig_bool_label)




ax.axhline(y=0, lw=2.5, ls=':', color='k', zorder=1)
#min_x, ma


ax.set_xlabel("Oscillation timescale", fontsize=12)
ax.set_ylabel("GAM coefficient for %s" % env_variable_to_plot, fontsize=12)
#ax.axvline(x=focal_otu_sine_param, lw=2.5, ls=':', label='OTU1', color='k', zorder=1)

#ax.hlines(y=focal_otu_coeff, xmin=min(), xmax=1.0, color='b')


ax.legend(loc='upper left', fontsize=6)


fig.subplots_adjust(hspace=0.35, wspace=0.25)
fig_name = "%scompare_otu1_param_%s.png" % (config.analysis_directory, env_variable_to_plot)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


