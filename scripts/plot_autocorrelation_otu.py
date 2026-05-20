import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
from scipy.special import polygamma, iv
# numdifftools also installed
import pickle
import tsdata_to_cpsd
from scipy.interpolate import interp1d
import plot_cpsd_lag
import sine_parameter_utils


n = plot_cpsd_lag.n
# frequency resolution
fres = plot_cpsd_lag.fres
h = plot_cpsd_lag.h
fs = plot_cpsd_lag.fs
nfft = plot_cpsd_lag.nfft
window = plot_cpsd_lag.window

noverlap = plot_cpsd_lag.noverlap
min_coh_xy = plot_cpsd_lag.min_coh_xy
n_surr = plot_cpsd_lag.noverlap




autocorrelation_dict_path = config.data_directory + 'autocorrelation_dict.pickle'
taxonomy_dict = utils.build_taxonomy_dict()

numpy.seterr(divide='ignore', invalid='ignore')
min_n_obs = 10


#def autocorrelation(tau, delta_t):

#label_dict = {'DNA':  r'$R_{\tilde{X}_{i}}(\Delta t)$'}




def calculate_autocorrelation_gamma_time_varying_mean(delta_t, A_i, tau_i, sigma_i):

    # predicted autocorrelation of ln x_i(t)
    delta_t = numpy.asarray(delta_t)

    # shape parameter of Gamma distribution
    beta_i = (2 - sigma_i) / sigma_i

    # trigamma function: variance of ln x for Gamma(beta_i)
    trigamma_beta = polygamma(1, beta_i)

    # signal and noise variance
    signal_var = A_i**2 / 2
    noise_var  = trigamma_beta
    total_var  = signal_var + noise_var

    # attenuated cosine (valid for delta_t != 0)
    rho = (signal_var / total_var) * numpy.cos(2 * numpy.pi * delta_t / tau_i)

    # enforce rho(0) = 1
    rho[delta_t == 0] = 1.0

    return rho



def calculate_autocorrelation_gamma_time_varying_mean_ratio(delta_t, amp_rna, amp_dna, tau_rna, tau_dna, beta_rna, beta_dna, C_rd_func=None, C_rd_0=None):
    
    numerator = ((amp_rna**2 / 2) * numpy.cos(2 * numpy.pi * delta_t / tau_rna) + (amp_dna**2 / 2) * numpy.cos(2 * numpy.pi * delta_t / tau_dna))

    psi_sum = polygamma(1, beta_rna) + polygamma(1, beta_dna)

    denominator = ((amp_rna**2 + amp_dna**2) / 2 + psi_sum)

    if (C_rd_func is not None) and (C_rd_0 is not None):
        C_rd_at_delta_t = C_rd_func(delta_t)
        numerator   = numerator - 2 * C_rd_at_delta_t
        denominator = denominator - 2 * C_rd_0

    
    return numerator / denominator



def cross_covariance_from_cpsd(S, i, j, nfft, fs):

    # Extract cross-covariance C_rd(delta_t) from one-sided CPSD matrix S
    # S is (n, n, h), output of cpsd_welch_matlab
    # i, j channel indices for the cross-covariance
    
    # Returns
    # lags, array, time lags
    # C_rd array, real cross-covariance at each lag

    # complex, shape (h,) where h = nfft//2 + 1
    one_sided = S[i, j, :]

    # Reconstruct two-sided spectrum
    # DC and Nyquist are not mirrored; interior bins are
    two_sided = numpy.zeros(nfft, dtype=complex)
     # DC
    two_sided[0]  = one_sided[0]
    # positive freqs
    two_sided[1:nfft//2] = one_sided[1:nfft//2]
    # Nyquist
    two_sided[nfft//2] = one_sided[nfft//2]
    # negative freqs
    two_sided[nfft//2+1:] = numpy.conj(one_sided[nfft//2-1:0:-1])

    # IFFT and scale by fs to convert density => covariance
    c_rd = numpy.real(numpy.fft.ifft(two_sided)) * fs

    dt   = 1.0 / fs
    lags = numpy.arange(nfft) * dt

    return lags, c_rd



def make_cross_covariance_interpolator(S, i, j, nfft, fs):
    
    # Returns callable C_rd(delta_t) and C_rd(0) for use in autocorrelation

    lags, C_rd = cross_covariance_from_cpsd(S, i, j, nfft, fs)
    interpolator = interp1d(lags, C_rd, bounds_error=False, fill_value=0.0)
    return interpolator, C_rd[0]




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

    #calculate_autocorrelation_gamma_time_varying_mean
    
    autocorr_dict = {}
    autocorr_dict['env'] = {}
    autocorr_dict['env']['water_temp'] = {}
    autocorr_dict['env']['water_temp']['delta_t_env'] = delta_t_env.tolist()
    autocorr_dict['env']['water_temp']['autocorr_obs_env'] = autocorr_obs_env.tolist()
    autocorr_dict['otu'] = {}

    idx_all = list(range(len(param_dict['otu_labels'])))
    #err_all = []
    #iv_all = []
    for otu_i_idx in idx_all:

        otu_i = param_dict['otu_labels'][otu_i_idx]

        autocorr_dict['otu'][otu_i] = {}

        for data_type in ['RNA', 'DNA']:

            #param_mean_i = param_dict['param_mean_mle'][data_type][otu_i_idx]
            amp_i = param_dict['amp_mle'][data_type][otu_i_idx]
            tau_i = 2*numpy.pi/param_dict['freq_mle'][data_type][otu_i_idx]

            sigma_i = param_dict['sigma_corrected'][data_type][otu_i_idx]

            afd_i = param_dict['data']['clr_afd'][data_type][otu_i_idx]
            days_i = param_dict['data']['days'][data_type][otu_i_idx]

            afd_i = numpy.asarray(afd_i)
            days_i = numpy.asarray(days_i)
            
            #afd_i_rescaled = (afd_i - param_mean_leastsq_i)/amp_leastsq_i

            autocorr_obs_i, delta_t_i, n_i = utils.calculate_autocorrelation(afd_i, days_i)
            
            delta_t_inter = numpy.intersect1d(delta_t_i, delta_t_env)

            delta_t_i_to_keep_idx = [numpy.where(delta_t_i==t)[0][0] for t in delta_t_inter]
            delta_t_env_to_keep_idx = [numpy.where(delta_t_env==t)[0][0] for t in delta_t_inter]

            rho_autocorr_clr_vs_temp = numpy.corrcoef(autocorr_obs_i[delta_t_i_to_keep_idx], autocorr_obs_env[delta_t_env_to_keep_idx])[0,1]

            autocorr_pred_i = calculate_autocorrelation_gamma_time_varying_mean(delta_t_i, amp_i, tau_i, sigma_i)
            #err_i = numpy.mean(numpy.sqrt((autocorr_pred_i - autocorr_obs_i)**2))
            #iv_i = numpy.log(iv(0, amp_i))

            #err_all.append(err_i)
            #iv_all.append(iv_i)

            #print(sigma_i, iv_i, err_i)
            autocorr_dict['otu'][otu_i][data_type] = {}
            autocorr_dict['otu'][otu_i][data_type]['delta_t'] = delta_t_i
            autocorr_dict['otu'][otu_i][data_type]['autocorr_obs'] = autocorr_obs_i
            autocorr_dict['otu'][otu_i][data_type]['autocorr_pred'] = autocorr_pred_i
            autocorr_dict['otu'][otu_i][data_type]['rho_autocorr_clr_vs_temp'] = rho_autocorr_clr_vs_temp


        # RNA:DNA
        afd_rna_i = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_i_idx])
        afd_dna_i = numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_i_idx])
        afd_ratio_i = afd_rna_i - afd_dna_i

        days_ratio_i = numpy.asarray(param_dict['data']['days']['RNA'][otu_i_idx])
        autocorr_obs_ratio_i, delta_t_ratio_i, n_ratio_i = utils.calculate_autocorrelation(afd_ratio_i, days_ratio_i)
            
        amp_rna_i = param_dict['amp_mle']['RNA'][otu_i_idx]
        amp_dna_i = param_dict['amp_mle']['DNA'][otu_i_idx]

        tau_rna_i = 2*numpy.pi/param_dict['freq_mle']['RNA'][otu_i_idx]
        tau_dna_i = 2*numpy.pi/param_dict['freq_mle']['DNA'][otu_i_idx]

        beta_rna_i = param_dict['beta_corrected']['RNA'][otu_i_idx]
        beta_dna_i = param_dict['beta_corrected']['DNA'][otu_i_idx]

        autocorr_pred_ratio_i = calculate_autocorrelation_gamma_time_varying_mean_ratio(delta_t_ratio_i, amp_rna_i, amp_dna_i, tau_rna_i, tau_dna_i, beta_rna_i, beta_dna_i)

        autocorr_dict['otu'][otu_i]['ratio'] = {}
        autocorr_dict['otu'][otu_i]['ratio']['delta_t'] = delta_t_ratio_i
        autocorr_dict['otu'][otu_i]['ratio']['autocorr_obs'] = autocorr_obs_ratio_i
        autocorr_dict['otu'][otu_i]['ratio']['autocorr_pred'] = autocorr_pred_ratio_i

        #rna_dna_i = numpy.column_stack((afd_rna_i, afd_dna_i))

        #S = tsdata_to_cpsd.cpsd_welch_matlab(rna_dna_i, n=n, h=h, nfft=nfft, window=window, noverlap=noverlap, fs=fs)
        #C_rd_func, C_rd_0 = make_cross_covariance_interpolator(S, i=0, j=1, nfft=nfft, fs=fs)

        #autocorr_pred_ratio_cov_i = calculate_autocorrelation_gamma_time_varying_mean_ratio(delta_t_ratio_i, amp_rna_i, amp_dna_i, tau_rna_i, tau_dna_i, beta_rna_i, beta_dna_i, C_rd_func, C_rd_0)
        #autocorr_dict['otu'][otu_i]['ratio']['autocorr_pred_cov'] = autocorr_pred_ratio_cov_i


    sys.stderr.write("Saving correlation dictionary...\n")
    with open(autocorrelation_dict_path, 'wb') as outfile:
        pickle.dump(autocorr_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def plot_autocorrelation_otu(data_type):

    #param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
    autocorr_dict = pickle.load(open(autocorrelation_dict_path, "rb"))

    otu_labels = list(autocorr_dict['otu'].keys())
    #delta_t_env = autocorr_dict['env']['water_temp']['delta_t_env']
    #autocorr_obs_env = autocorr_dict['env']['water_temp']['autocorr_obs_env']

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    asv_count = 0
    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))
            
            delta_t_c = autocorr_dict['otu'][otu_labels[asv_count]][data_type]['delta_t']
            autocorr_obs_c = autocorr_dict['otu'][otu_labels[asv_count]][data_type]['autocorr_obs']
            autocorr_pred_c = autocorr_dict['otu'][otu_labels[asv_count]][data_type]['autocorr_pred']
            #autocorr_pred_c = 0.5*numpy.cos((delta_t_c*param_dict['freq_mle'][data_type][c]))
            
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



#def plot_autocorr_diagnosis():

def plot_autocorrelation_ratio_otu(cov=False):

    autocorr_dict = pickle.load(open(autocorrelation_dict_path, "rb"))

    otu_labels = list(autocorr_dict['otu'].keys())

    fig_label = ''
    pred_type = 'autocorr_pred'
    if cov == True:
        fig_label = '_cov'
        pred_type = 'autocorr_pred_cov'

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)
    idx_all = list(range(len(otu_labels)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    asv_count = 0
    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))
            
            delta_t_c = autocorr_dict['otu'][otu_labels[asv_count]]['ratio']['delta_t']
            autocorr_obs_c = autocorr_dict['otu'][otu_labels[asv_count]]['ratio']['autocorr_obs']
            autocorr_pred_c = autocorr_dict['otu'][otu_labels[asv_count]]['ratio'][pred_type]
            
            ax.scatter(delta_t_c, autocorr_obs_c, s=7, alpha=1, zorder=1, c=utils.dna_rna_color_dict['ratio'], label='Observed')
            ax.plot(delta_t_c, autocorr_pred_c, ls='-', lw=3, zorder=2, c=utils.dna_rna_color_dict['ratio'], label='Predicted')

            ax.set_xlabel("Time difference (days), " + r'$\Delta t$', fontsize = 10)
            ax.set_ylabel("Autocorrelation, " + utils.sample_label_dict['ratio'], fontsize = 10)
            ax.set_title('ASV %d (%s)' % (asv_count + 1, taxonomy_dict[otu_labels[asv_count]]['family']), fontsize=12)



            if (chunk_idx==0) and (c_idx==0):
                ax.legend(loc='upper right', fontsize=8)
    
            asv_count += 1


    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%sautocorrelation_otu_ratio%s.png" % (config.analysis_directory, fig_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





if __name__ == "__main__":

    #  ['DNA', 'RNA', 'ratio']

    parser = argparse.ArgumentParser(description='Variable to plot')
    parser.add_argument('-d', '--data_type', type=str, required=False,
                        help='Data type to plot: RNA, DNA or ratio')

    args = parser.parse_args()    

    make_autocorrelation_dict()

    #plot_autocorrelation_otu(args.data_type)

    plot_autocorrelation_ratio_otu(cov=False)

    

    