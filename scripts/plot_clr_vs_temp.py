
import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats, signal
# numdifftools also installed
import pickle
import sine_parameter_utils


clr_status=True



s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples, min_occupancy=1)
rel_s_by_s_dna_rescaled = (rel_s_by_s_dna.T - numpy.mean(rel_s_by_s_dna, axis=1)).T
rel_s_by_s_rna_rescaled = (rel_s_by_s_rna.T - numpy.mean(rel_s_by_s_rna, axis=1)).T

rel_s_by_s_ratio = rel_s_by_s_rna - rel_s_by_s_dna
rel_s_by_s_ratio_rescaled = (rel_s_by_s_ratio.T - numpy.mean(rel_s_by_s_ratio, axis=1)).T


# get days
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
env_variable_array = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples[(sample_type=='RNA')]])
days_env = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


#data_type = 'ratio'


def make_time_delay_dict(min_n_obs=10):

    afd = rel_s_by_s_dna_rescaled[0,:]

    to_keep_idx = (~numpy.isnan(afd)) & (~numpy.isnan(env_variable_array))

    afd_to_keep = afd[to_keep_idx]
    env_variable_array_to_keep = env_variable_array[to_keep_idx]
    days_env_to_keep = days_env[to_keep_idx]

    afd_to_keep_standard = (afd_to_keep - numpy.mean(afd_to_keep))/numpy.std(afd_to_keep)
    env_variable_array_to_keep_standard = (env_variable_array_to_keep - numpy.mean(env_variable_array_to_keep))/numpy.std(env_variable_array_to_keep)

    time_delay_corr = signal.correlate(afd_to_keep_standard, env_variable_array_to_keep_standard, mode="same")
    lags = signal.correlation_lags(len(afd_to_keep), len(env_variable_array_to_keep), mode="same")
    #lags = signal.correlation_lags(days_env_to_keep, days_env_to_keep, mode="full")
    #correlation = signal.correlate(afd_to_keep, env_variable_array, mode="full")

    #print(time_delay_corr.shape, env_variable_array_to_keep.shape)
    #print(lags)

    #print(time_delay_corr[lags==0])
    print(time_delay_corr)
    #print(numpy.corrcoef(afd_to_keep, env_variable_array_to_keep)[0,1])



def calculate_delay_corr(array_1, array_2, t, min_n_obs=10):

    delta_t_idx_range = range(0, len(t) - min_n_obs - 1)

    delta_t_days_all = []
    rho_all = []

    for delta_t_idx in delta_t_idx_range:

        if delta_t_idx > 0:

            array_1_fwd = array_1[delta_t_idx:]
            array_2_fwd = array_2[:int(-1*delta_t_idx)]

            array_1_rev = array_1[:int(-1*delta_t_idx)]
            array_2_rev = array_2[delta_t_idx:]
     

            delta_t_days_fwd = t[delta_t_idx] - t[0]
            #delta_t_days_rev = t[0] - t[delta_t_idx]

            delta_t_days_all.append(delta_t_days_fwd)
            # rev
            delta_t_days_all.append(int(-1*delta_t_days_fwd))

            rho_fwd = numpy.corrcoef(array_1_fwd, array_2_fwd)[0,1]
            rho_rev = numpy.corrcoef(array_1_rev, array_2_rev)[0,1]

            rho_all.append(rho_fwd)
            rho_all.append(rho_rev)



    delta_t_days_all = numpy.asarray(delta_t_days_all)
    rho_all = numpy.asarray(rho_all)

    delta_t_days_sort_idx = numpy.argsort(delta_t_days_all)

    delta_t_days_all = delta_t_days_all[delta_t_days_sort_idx]
    rho_all = rho_all[delta_t_days_sort_idx]

    return delta_t_days_all, rho_all







def make_plot(data_type):

    idx_all = list(range(rel_s_by_s_rna_rescaled.shape[0]))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):
            
            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            if data_type == 'DNA':
                rel_s_by_s_data_type = rel_s_by_s_dna_rescaled

            elif data_type == 'RNA':
                rel_s_by_s_data_type = rel_s_by_s_rna_rescaled

            else:
                rel_s_by_s_data_type = rel_s_by_s_ratio_rescaled
            
            afd = rel_s_by_s_data_type[c,:]

            ax.scatter(1/env_variable_array, afd, s=8, alpha=0.5, c=utils.dna_rna_color_dict[data_type])
            ax.set_xscale('log', basex=10)
            ax.set_title(otu_labels_subset[c], fontsize=11)
            ax.set_xlabel("Inverse temperature, " + r'$T^{-1}$', fontsize = 10)
            ax.set_ylabel("CLR transformed abundance, %s" % utils.rescaled_label_clr_dict[data_type], fontsize = 10)



    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%sclr_vs_temp_%s.png" % (config.analysis_directory, data_type)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_time_delay_env_rho():

    idx_all = list(range(rel_s_by_s_rna_rescaled.shape[0]))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):
            
            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            for data_type in utils.data_type_all:

                if data_type == 'DNA':
                    rel_s_by_s_data_type = rel_s_by_s_dna_rescaled

                elif data_type == 'RNA':
                    rel_s_by_s_data_type = rel_s_by_s_rna_rescaled

                else:
                    rel_s_by_s_data_type = rel_s_by_s_ratio_rescaled

                afd = rel_s_by_s_data_type[c,:]

                to_keep_idx = (~numpy.isnan(afd)) & (~numpy.isnan(env_variable_array))
                afd_to_keep = afd[to_keep_idx]
                env_variable_array_to_keep = env_variable_array[to_keep_idx]
                days_env_to_keep = days_env[to_keep_idx]

                delta_t_days_all, rho_all = calculate_delay_corr(afd_to_keep, env_variable_array_to_keep, days_env_to_keep)
                ax.scatter(delta_t_days_all, rho_all, s=8, alpha=0.5, c=utils.dna_rna_color_dict[data_type])

                ax.set_title(otu_labels_subset[c], fontsize=11)
                ax.set_xlabel("Time delay (days), " + r'$\Delta t$', fontsize = 10)
                ax.set_ylabel("Time-shifted corr bw CLR abund. &\nwater temp., " + r'$\rho( c(n(t + \Delta t)), T(t) )$', fontsize = 10)



    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_delay_env_rho.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()


plot_time_delay_env_rho()

#make_plot('DNA')
#make_plot('RNA')
#make_plot('ratio')


#
#make_time_delay_dict(rel_s_by_s_dna_rescaled[0,:], env_variable_array, )
