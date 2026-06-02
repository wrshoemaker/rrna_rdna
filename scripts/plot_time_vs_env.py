import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats, signal
import pickle
import sine_parameter_utils


env_variable_all_nested = [['water_temp', 'specific_conductivity', 'dissolved_oxygen'], ['salinity', 'secchi_depth', 'ph'], ['total_nitrogen', 'total_phosphorus', 'doc']]

env_variable = ['doc', 'secchi_depth', 'ph', 'dissolved_oxygen', 'water_temp', 'salinity', 'total_nitrogen', 'specific_conductivity', 'total_phosphorus']


metadata_dict = utils.build_metadata_dict()
s_by_s, otu_labels, samples = utils.load_count_data()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])
day_of_year = numpy.asarray([metadata_dict[s]['day_of_year'] for s in samples[(sample_type=='RNA')]])

minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


def plot_time_vs_env():

    param_env_dict = sine_parameter_utils.load_param_env_dict()    
    
    fig = plt.figure(figsize = (5, 27))
    #fig.subplots_adjust(bottom= 0.15)
    fig.subplots_adjust(hspace=0)

    for env_variable_j_idx, env_variable_j in enumerate(env_variable):

        ax = plt.subplot2grid((9, 1), (env_variable_j_idx, 0), colspan=1)

        env_variable_array = numpy.asarray([metadata_dict[s][env_variable_j] for s in samples[(sample_type=='RNA')]])
        # remove nans
        env_to_keep_idx = (~numpy.isnan(env_variable_array))
        env_variable_array_clean = env_variable_array[env_to_keep_idx]
        days_clean = days[env_to_keep_idx]
        env_variable_dict_idx = param_env_dict['env_variables_labels'].index(env_variable_j)
        

        ax.scatter(days_clean, env_variable_array_clean, s=8, alpha=1, zorder=2, c='k')
        #ax.set_yscale('log', basey=10)
        #ax.tick_params(axis='both', labelsize=7)

        days_range = numpy.linspace(min(days_clean), max(days_clean), 1000)

        #print(env_variable_j, numpy.pi*2/param_env_dict['freq_leastsq'][env_variable_dict_idx])

        sine_prediction = param_env_dict['amp_leastsq'][env_variable_dict_idx] * numpy.sin(param_env_dict['freq_leastsq'][env_variable_dict_idx] * days_range + param_env_dict['phase_leastsq'][env_variable_dict_idx]) + param_env_dict['param_mean_leastsq'][env_variable_dict_idx]
        ax.plot(days_range, sine_prediction, lw=3, ls='-', alpha=0.9, c='k', zorder=1, label='Sine function')
        ax.set_ylabel(utils.env_variable_label_dict[env_variable_j], fontsize=12)
        ax.set_xlim([0, max(days_clean)])
        ax.set_xticks(minor_days, minor=True)
        ax.set_xticks(major_days, minor=False)
        if env_variable_j_idx == len(env_variable)-1:
            
            ax.set_xlabel('Time (days)', fontsize=14)
            ax.set_xticklabels(major_labels, minor=False, fontsize=6)


        ax.yaxis.set_tick_params(labelsize=6)
        ax.xaxis.set_tick_params(labelsize=6)
            

        #if (env_variable_j_idx == 0) and (nested_i_idx==0):
        #    ax.legend(loc='upper right', fontsize=7)



    #fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_env_stacked.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()

    # split 1
    fig = plt.figure(figsize = (5, 15))
    #fig.subplots_adjust(bottom= 0.15)
    fig.subplots_adjust(hspace=0)

    for env_variable_j_idx, env_variable_j in enumerate(env_variable[:5]):

        ax = plt.subplot2grid((9, 1), (env_variable_j_idx, 0), colspan=1)

        env_variable_array = numpy.asarray([metadata_dict[s][env_variable_j] for s in samples[(sample_type=='RNA')]])
        # remove nans
        env_to_keep_idx = (~numpy.isnan(env_variable_array))
        env_variable_array_clean = env_variable_array[env_to_keep_idx]
        days_clean = days[env_to_keep_idx]
        env_variable_dict_idx = param_env_dict['env_variables_labels'].index(env_variable_j)
        
        ax.scatter(days_clean, env_variable_array_clean, s=8, alpha=1, zorder=2, c='k')
        #ax.set_yscale('log', basey=10)
        #ax.tick_params(axis='both', labelsize=7)

        days_range = numpy.linspace(min(days_clean), max(days_clean), 1000)

        #print(env_variable_j, numpy.pi*2/param_env_dict['freq_leastsq'][env_variable_dict_idx])

        sine_prediction = param_env_dict['amp_leastsq'][env_variable_dict_idx] * numpy.sin(param_env_dict['freq_leastsq'][env_variable_dict_idx] * days_range + param_env_dict['phase_leastsq'][env_variable_dict_idx]) + param_env_dict['param_mean_leastsq'][env_variable_dict_idx]
        ax.plot(days_range, sine_prediction, lw=2, ls='-', alpha=0.9, c='k', zorder=1, label='Sine function')
        ax.set_ylabel(utils.env_variable_label_dict[env_variable_j], fontsize=12)
        ax.set_xlim([0, max(days_clean)])
        ax.set_xticks(minor_days, minor=True)
        ax.set_xticks(major_days, minor=False)
        if env_variable_j_idx == len(env_variable[:5])-1:
            
            ax.set_xlabel('Time (days)', fontsize=14)
            ax.set_xticklabels(major_labels, minor=False, fontsize=6)


        ax.yaxis.set_tick_params(labelsize=6)
        ax.xaxis.set_tick_params(labelsize=6)
            

        #if (env_variable_j_idx == 0) and (nested_i_idx==0):
        #    ax.legend(loc='upper right', fontsize=7)

    #fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_env_stacked_1.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



    # split 2
    fig = plt.figure(figsize = (5, 12))
    #fig.subplots_adjust(bottom= 0.15)
    fig.subplots_adjust(hspace=0)

    for env_variable_j_idx, env_variable_j in enumerate(env_variable[5:]):

        ax = plt.subplot2grid((9, 1), (env_variable_j_idx, 0), colspan=1)

        env_variable_array = numpy.asarray([metadata_dict[s][env_variable_j] for s in samples[(sample_type=='RNA')]])
        # remove nans
        env_to_keep_idx = (~numpy.isnan(env_variable_array))
        env_variable_array_clean = env_variable_array[env_to_keep_idx]
        days_clean = days[env_to_keep_idx]
        env_variable_dict_idx = param_env_dict['env_variables_labels'].index(env_variable_j)
        
        ax.scatter(days_clean, env_variable_array_clean, s=8, alpha=1, zorder=2, c='k')
        #ax.set_yscale('log', basey=10)
        #ax.tick_params(axis='both', labelsize=7)

        days_range = numpy.linspace(min(days_clean), max(days_clean), 1000)

        #print(env_variable_j, numpy.pi*2/param_env_dict['freq_leastsq'][env_variable_dict_idx])

        sine_prediction = param_env_dict['amp_leastsq'][env_variable_dict_idx] * numpy.sin(param_env_dict['freq_leastsq'][env_variable_dict_idx] * days_range + param_env_dict['phase_leastsq'][env_variable_dict_idx]) + param_env_dict['param_mean_leastsq'][env_variable_dict_idx]
        ax.plot(days_range, sine_prediction, lw=2, ls='-', alpha=0.9, c='k', zorder=1, label='Sine function')
        ax.set_ylabel(utils.env_variable_label_dict[env_variable_j], fontsize=12)
        ax.set_xlim([0, max(days_clean)])
        ax.set_xticks(minor_days, minor=True)
        ax.set_xticks(major_days, minor=False)
        if env_variable_j_idx == len(env_variable[5:])-1:
            
            ax.set_xlabel('Time (days)', fontsize=14)
            ax.set_xticklabels(major_labels, minor=False, fontsize=6)


        ax.yaxis.set_tick_params(labelsize=6)
        ax.xaxis.set_tick_params(labelsize=6)
            

        #if (env_variable_j_idx == 0) and (nested_i_idx==0):
        #    ax.legend(loc='upper right', fontsize=7)

    #fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_env_stacked_2.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





plot_time_vs_env()