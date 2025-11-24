import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
# numdifftools also installed
import pickle

import sine_parameter_utils



param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
#autocorr_dict = pickle.load(open(autocorrelation_dict_path, "rb"))

#otu_labels = list(autocorr_dict['otu'].keys())
#otu_labels.sort()


#delta_t_env = autocorr_dict['env']['water_temp']['delta_t_env']
#autocorr_obs_env = autocorr_dict['env']['water_temp']['autocorr_obs_env']

fig = plt.figure(figsize = (20, 20))
fig.subplots_adjust(bottom= 0.15)

s_by_s, otu_labels, samples = utils.load_count_data()

idx_all = list(range(len(param_dict['data']['clr_afd']['RNA'])))
chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

for chunk_idx, chunk in enumerate(chunk_all):

    for c_idx, c in enumerate(chunk):

        ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

        for data_type in ['RNA', 'DNA']:

            afd_i = numpy.asarray(param_dict['data']['clr_afd'][data_type][c])
            days_i = numpy.asarray(param_dict['data']['days'][data_type][c])

            logfold_i = (afd_i[1:] - afd_i[:-1]) / (days_i[1:] - days_i[:-1])
            days_logfold_i = days_i[:-1]

            otu_label_i = otu_labels[c]

            rho_all, delta_t_all, n_all = utils.calculate_autocorrelation(logfold_i, days_logfold_i, min_n_obs=10)

            #if otu_label_i == 'Otu000001':
            #    print(days_i)
            #    print(n_all)

            print(delta_t_all[0], rho_all[0])

            ax.scatter(delta_t_all, rho_all, s=7, alpha=0.7, zorder=2, c=utils.dna_rna_color_dict[data_type], label='Observed')
            #ax.plot(delta_t_all, rho_all, ls='-', lw=3, zorder=2, c=utils.dna_rna_color_dict[data_type], label='Predicted')
            ax.plot(delta_t_all, rho_all, ls='-', lw=1, alpha=0.6, zorder=1, c=utils.dna_rna_color_dict[data_type], label='Predicted')

            ax.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 10)
            ax.set_ylabel("Autocorr. of log-fold growth", fontsize = 10)
            ax.set_title(otu_labels[c], fontsize=11)

            #if (chunk_idx==0) and (c_idx==0):
            #    ax.legend(loc='upper right', fontsize=8)



        ax.axhline(y=0, ls=':', lw=2, c='k', zorder=3, label='Data')



fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%sautocorr_logfold_otu.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
