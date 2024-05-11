import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')



numpy.random.seed(123456789)


s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)
# s_by_s.shape = (246, 134265)

metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])


sample_type_rna_idx = (sample_type=='RNA')
sample_type_dna_idx = (sample_type=='DNA')

sample_type_rna = samples[sample_type_rna_idx]

rel_s_by_s_rna = rel_s_by_s[:,sample_type_rna_idx]
rel_s_by_s_dna = rel_s_by_s[:,sample_type_dna_idx]


days = numpy.asarray([metadata_dict[s]['day'] for s in sample_type_rna])

occupancy_rna = numpy.sum((rel_s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
occupancy_dna = numpy.sum((rel_s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)

subset_idx = (occupancy_rna==1) & (occupancy_dna==1)

rel_s_by_s_rna_subset = rel_s_by_s_rna[subset_idx,:]
rel_s_by_s_dna_subset = rel_s_by_s_dna[subset_idx,:]

afd_ratio_all = [rel_s_by_s_rna_subset[i_idx,:]/rel_s_by_s_dna_subset[i_idx,:] for i_idx in range(sum(subset_idx))]
afd_ratio_all = numpy.asarray(afd_ratio_all)

mean_afd_ratio_all = numpy.mean(afd_ratio_all, axis=1)
#std_afd_ratio_all = numpy.std(afd_ratio_all, axis=0)
rescaled_afd_ratio_all = (afd_ratio_all.T / mean_afd_ratio_all).T

#mean_log_rescaled_afd_ratio_all = numpy.mean(numpy.log10(rescaled_afd_ratio_all), axis=0)
mean_rescaled_afd_ratio_all = numpy.mean(rescaled_afd_ratio_all, axis=0)
#mean_rescaled_afd_ratio_all = 10**numpy.mean(numpy.log10(rescaled_afd_ratio_all), axis=0)


# environmental analysis....
env_variable_all = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph', 'air_temperature']
env_variable_dict = {}
for env_variable in env_variable_all:
    env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in sample_type_rna])
    env_variable_dict[env_variable] = env_variable_array



for env_pair in itertools.combinations(env_variable_all, 2):

    env_variable_array_1 = env_variable_dict[env_pair[0]]
    env_variable_array_2 = env_variable_dict[env_pair[1]]

    to_keep_idx = numpy.isfinite(env_variable_array_1) & numpy.isfinite(env_variable_array_2)
    rho = numpy.corrcoef(env_variable_array_1[to_keep_idx], env_variable_array_2[to_keep_idx])[0,1]
    

env_variable_all_nested = [['water_temp', 'specific_conductivity'], ['dissolved_oxygen', 'salinity'], ['secchi_depth', 'ph']]


fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(bottom= 0.15)

for nested_i_idx, nested_i in enumerate(env_variable_all_nested):

    for env_variable_j_idx, env_variable_j in enumerate(nested_i):

        ax = plt.subplot2grid((3, 2), (nested_i_idx, env_variable_j_idx), colspan=1)

        ax.scatter(days, mean_rescaled_afd_ratio_all, s=5, alpha=1, zorder=2, c='k')
        ax.plot(days, mean_rescaled_afd_ratio_all, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
        ax.set_yscale('log', basey=10)
        ax.tick_params(axis='both', labelsize=7)
        ax.set_ylabel('Mean rescaled RNA/DNA ratio', c='k', fontsize=8)


        env_variable_j_array = env_variable_dict[env_variable_j]
        env_variable_to_keep_idx = numpy.isfinite(env_variable_j_array)
        days_env_variable_to_keep = days[env_variable_to_keep_idx]
        env_variable_j_array_to_keep = env_variable_j_array[env_variable_to_keep_idx]


        ax_env = ax.twinx()
        ax_env.scatter(days_env_variable_to_keep, env_variable_j_array_to_keep, s=5, alpha=1, zorder=2, c='r')
        ax_env.plot(days_env_variable_to_keep, env_variable_j_array_to_keep, lw=1, ls='-', alpha=0.5, c='r', zorder=1)
        ax_env.tick_params(axis='both', labelsize=7)
        ax_env.set_ylabel('Environmental variable', c='r', fontsize=8)

        # correlation
        log_mean_rescaled_afd_ratio_all = numpy.log10(mean_rescaled_afd_ratio_all)
        rho_to_keep_idx = (numpy.isfinite(env_variable_j_array) & numpy.isfinite(log_mean_rescaled_afd_ratio_all))
        rho = numpy.corrcoef(env_variable_j_array[rho_to_keep_idx], log_mean_rescaled_afd_ratio_all[rho_to_keep_idx])[0,1]
        print(rho)

        #if env_variable_j == 'water_temp':

        #    air_temp = env_variable_dict['air_temperature']
        #    air_temp_to_keep_idx = numpy.isfinite(air_temp)
        #    days_air_temp_to_keep = days[air_temp_to_keep_idx]
        #    air_temp_to_keep = air_temp[air_temp_to_keep_idx]
                  
        #    ax_env.scatter(days_air_temp_to_keep, air_temp_to_keep, s=5, alpha=1, zorder=2, c='b')
        #    ax_env.plot(days_air_temp_to_keep, air_temp_to_keep, lw=1, ls='-', alpha=0.5, c='b', zorder=1)

        #    ax.set_title("Temperature (°C), " + r'$\rho^{2} = $' + str(round(rho**2, 3)), fontsize=10)


        #else:
        ax.set_title(utils.env_variable_label_dict[env_variable_j] + ', ' + r'$\rho^{2} = $' + str(round(rho**2, 3)) , fontsize=10)

            
        


        



fig.subplots_adjust(hspace=0.35,wspace=0.4)
fig_name = "%srescaled_vs_environment.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


