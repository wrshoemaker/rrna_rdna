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



autocorrelation_dict_path = config.data_directory + 'autocorrelation_dict.pickle'
taxonomy_dict = utils.build_taxonomy_dict()

numpy.seterr(divide='ignore', invalid='ignore')
min_n_obs = 10


#def autocorrelation(tau, delta_t):

#label_dict = {'DNA':  r'$R_{\tilde{X}_{i}}(\Delta t)$'}


def make_autocorrelation_dict():

    metadata_dict = utils.build_metadata_dict()

    s_by_s, otu_labels, samples = utils.load_count_data()

    param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))

    param_env_dict = sine_parameter_utils.load_param_env_dict()
    # get days
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    env_variable_array = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples[(sample_type=='RNA')]])
    days_env = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

    to_keep_idx = ~numpy.isnan(env_variable_array)
    env_variable_array = env_variable_array[to_keep_idx]
    days_env = days_env[to_keep_idx]

    env_variable_array_rescaled = (env_variable_array - param_env_dict['param_mean_leastsq'][0])/param_env_dict['amp_leastsq'][0]
    autocorr_obs_env, delta_t_env, n_env = utils.calculate_autocorrelation(env_variable_array_rescaled, days_env)

    
    autocorr_dict = {}
    autocorr_dict['env'] = {}
    autocorr_dict['env']['water_temp'] = {}
    autocorr_dict['env']['water_temp']['delta_t_env'] = delta_t_env.tolist()
    autocorr_dict['env']['water_temp']['autocorr_obs_env'] = autocorr_obs_env.tolist()

    autocorr_dict['otu'] = {}

    idx_all = list(range(len(param_dict['otu_labels'])))

    for otu_i_idx in idx_all:

        otu_i = param_dict['otu_labels'][otu_i_idx]

        autocorr_dict['otu'][otu_i] = {}

        for data_type in ['RNA', 'DNA']:

            param_mean_leastsq_i = param_dict['param_mean_mle'][data_type][otu_i_idx]
            amp_leastsq_i = param_dict['amp_mle'][data_type][otu_i_idx]

            afd_i = param_dict['data']['clr_afd'][data_type][otu_i_idx]
            days_i = param_dict['data']['days'][data_type][otu_i_idx]

            afd_i = numpy.asarray(afd_i)
            days_i = numpy.asarray(days_i)
            
            afd_i_rescaled = (afd_i - param_mean_leastsq_i)/amp_leastsq_i

            autocorr_obs_i, delta_t_i, n_i = utils.calculate_autocorrelation(afd_i_rescaled, days_i)

            delta_t_inter = numpy.intersect1d(delta_t_i, delta_t_env)

            delta_t_i_to_keep_idx = [numpy.where(delta_t_i==t)[0][0] for t in delta_t_inter]
            delta_t_env_to_keep_idx = [numpy.where(delta_t_env==t)[0][0] for t in delta_t_inter]

            rho_autocorr_clr_vs_temp = numpy.corrcoef(autocorr_obs_i[delta_t_i_to_keep_idx], autocorr_obs_env[delta_t_env_to_keep_idx])[0,1]

            autocorr_dict['otu'][otu_i][data_type] = {}
            autocorr_dict['otu'][otu_i][data_type]['delta_t'] = delta_t_i.tolist()
            autocorr_dict['otu'][otu_i][data_type]['autocorr_obs'] = autocorr_obs_i.tolist()
            autocorr_dict['otu'][otu_i][data_type]['rho_autocorr_clr_vs_temp'] = rho_autocorr_clr_vs_temp



    sys.stderr.write("Saving correlation dictionary...\n")
    with open(autocorrelation_dict_path, 'wb') as outfile:
        pickle.dump(autocorr_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def plot_autocorrelation_otu(data_type):

    param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
    autocorr_dict = pickle.load(open(autocorrelation_dict_path, "rb"))

    otu_labels = list(autocorr_dict['otu'].keys())
    delta_t_env = autocorr_dict['env']['water_temp']['delta_t_env']
    autocorr_obs_env = autocorr_dict['env']['water_temp']['autocorr_obs_env']

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    asv_count = 0
    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))


            delta_t_c = numpy.asarray(autocorr_dict['otu'][otu_labels[c]][data_type]['delta_t'])
            autocorr_obs_c = autocorr_dict['otu'][otu_labels[c]][data_type]['autocorr_obs']
            
            autocorr_pred_c = 0.5*numpy.cos((delta_t_c*param_dict['freq_mle'][data_type][c]))
            ax.scatter(delta_t_c, autocorr_obs_c, s=7, alpha=1, zorder=1, c=utils.dna_rna_color_dict[data_type], label='Observed')
            ax.plot(delta_t_c, autocorr_pred_c, ls='-', lw=3, zorder=2, c=utils.dna_rna_color_dict[data_type], label='Predicted')

            ax.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 10)
            ax.set_ylabel("Autocorrelation, " + utils.sample_label_dict[data_type], fontsize = 10)
            #ax.set_title(otu_labels[c], fontsize=11)
            ax.set_title('ASV %d (%s)' % (asv_count + 1, taxonomy_dict[otu_labels[asv_count]]['family']), fontsize=12)



            if (chunk_idx==0) and (c_idx==0):
                ax.legend(loc='upper right', fontsize=8)
    

            asv_count += 1


    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%sautocorrelation_otu_%s.png" % (config.analysis_directory, data_type)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()







if __name__ == "__main__":

    #  ['DNA', 'RNA', 'ratio']

    parser = argparse.ArgumentParser(description='Variable to plot')
    parser.add_argument('-d', '--data_type', type=str, required=False,
                        help='Data type to plot: RNA, DNA or ratio')

    args = parser.parse_args()    

    make_autocorrelation_dict()

    plot_autocorrelation_otu(args.data_type)

    

    