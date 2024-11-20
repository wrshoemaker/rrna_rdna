import config
import utils
import numpy
import sys
import pickle

from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.special import loggamma
import simulation_utils
import copy

from lmfit import Minimizer, create_params, fit_report, conf_interval


param_dict_path = config.data_directory + 'param_mle_dict.pickle'


# fsolve assumes func(x) = 0 
numpy.random.seed(123456789)

s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples, min_occupancy=1)

metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()


param_no_method_all = ['amp', 'freq', 'phase', 'param_mean']


def mle_sine_mean(params, *data):

    freq, phase, amp = params
    t, afd = data[0]

    k_0 = sum(afd*(numpy.exp( amp*numpy.sin(freq*t +phase) )**-1))/len(afd)
    x_bar = k_0*numpy.exp( amp*numpy.sin(freq*t +phase))

    mle_freq = sum(t*(afd/x_bar)*numpy.cos(freq*t + phase)) - sum(t*numpy.cos(freq*t + phase))
    mle_phase = sum((afd/x_bar)*numpy.cos(freq*t + phase)) - sum(numpy.cos(freq*t + phase))
    mle_amp = sum((afd/x_bar)*numpy.sin(freq*t + phase)) - sum(numpy.sin(freq*t + phase))

    mle_sum = mle_freq + mle_phase + mle_amp

    return mle_sum


def ll_sine_gamma(params, days_afd, afd):

    # minimize the negative log-likelihood 

    amp = params['amp']
    freq = params['freq']
    phase = params['phase']
    param_mean = params['param_mean']
    beta = params['beta']

    x_bar_pred = numpy.exp(amp*(numpy.cos(phase)*numpy.sin(freq*days_afd) + numpy.sin(phase)*numpy.cos(freq*days_afd))) * param_mean
    
    ll = (beta-1)*sum(numpy.log(afd)) - beta*sum(afd/x_bar_pred) - beta*sum(numpy.log(x_bar_pred)) + len(afd)*beta*numpy.log(beta) - len(afd)*loggamma(beta)

    return -1*ll


def second_rount_optimization(result_brute, fitter):

    # second round of optimization using least-squares with brute force as a starting point
    best_result_leastsq = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:
        #trial = fitter.minimize(method='emcee', params=candidate.params)
        trial = fitter.minimize(method='lbfgsb', params=candidate.params)
        if trial.chisqr < best_result_leastsq.chisqr:
            best_result_leastsq = trial
            #best_result_candidate = candidate

    return best_result_leastsq



def grid_search_mle_sine_wave(days_afd_, afd_, params_):

    # minimize the negative log-likelihood 
    fitter = Minimizer(ll_sine_gamma, params_, fcn_args=(days_afd_, afd_))
    
    # brute force results
    result_brute = fitter.minimize(method='brute', Ns=30, keep=25)

    return result_brute, fitter




def plot_exp_clr():

    fig = plt.figure(figsize = (8, 4))
    fig.subplots_adjust(bottom= 0.15)

    for c in range(2):

        ax = plt.subplot2grid((1, 2), (0, c))

        clr_afd = rel_s_by_s_dna[c,:]
        afd = numpy.exp(clr_afd)

        ax.scatter(days, afd, s=8, alpha=1, c=utils.dna_rna_color_dict['DNA'])
        #ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict['DNA'])
        ax.set_xlabel("Time (days)", fontsize=10)
        ax.set_ylabel('Exp. of CLR-transformed %s' % utils.rescaled_label_clr_dict['DNA'], fontsize=10)
        ax.set_title(otu_labels_subset[c], fontsize=11)

        #minor_days, major_days, major_labels
        ax.set_xlim([0, max(days)])
        ax.set_xticks(minor_days, minor=True)
        ax.set_xticks(major_days, minor=False)
        ax.set_xticklabels(major_labels, minor=False, fontsize=7)

        ax.set_ylim([min(afd), max(afd)])



    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stest_exp_clr.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()








def make_plot():

    #afd = rel_s_by_s_dna[1,:]

    param_dict = pickle.load(open(param_dict_path, "rb"))

    #params_fit = fsolve(mle_sine_mean, (2*numpy.pi/365, 0.3, 0.5), args=data)

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels_subset)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            clr_afd = rel_s_by_s_dna[c,:]
            #afd = numpy.exp(clr_afd)
            #data=[days, afd]


            #params_fit = minimize(mle_sine_mean, (2*numpy.pi/365, 0.3, 0.5), args=data, bounds=((2*numpy.pi/3000, 2*numpy.pi/30), (1e-3, 2*numpy.pi), (1e-2, 5)), method='L-BFGS-B')

            #freq, phase, amp = params_fit.x
            #param_mean = sum(afd*(numpy.exp( amp*numpy.sin(freq*days +phase) )**-1))/len(afd) 

            #print(params_fit.x)

            amp = param_dict['amp_leastsq'][c]
            freq = param_dict['freq_leastsq'][c]
            phase = param_dict['phase_leastsq'][c]
            param_mean = param_dict['param_mean_leastsq'][c]

            days_range = numpy.linspace(min(days), max(days), 1000)
            model_prediction = amp*numpy.sin(freq*days_range+phase) + numpy.log(param_mean)


            ax.scatter(days, clr_afd, s=8, alpha=1, c=utils.dna_rna_color_dict['DNA'])
            ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict['DNA'])
            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel(utils.rescaled_label_dict['DNA'], fontsize=10)
            ax.set_title(otu_labels_subset[c], fontsize=11)

            #minor_days, major_days, major_labels
            ax.set_xlim([0, max(days)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)
            ax.set_ylim([min(clr_afd), max(clr_afd)])


    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stest_mle_minimize.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def test_mle():

    s_by_s, otu_labels, samples = utils.load_count_data()
    rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples, min_occupancy=1)

    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])



    idx_all = list(range(len(otu_labels_subset)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    param_dict = {}
    param_dict['data'] = {}
    param_dict['data']['days'] = {}
    param_dict['data']['afd'] = {}

    param_dict['beta'] = []
    for p in param_no_method_all:

        # list for environmental variables because they only have one data_type
        param_dict['%s_brute' % p] = []
        param_dict['%s_mle' % p] = []

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            #if c != 4:
            #    continue 

            #ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            clr_afd = rel_s_by_s_dna[c,:]
            afd = numpy.exp(clr_afd)


            afd_to_keep_idx = (~numpy.isnan(afd))
            afd_clean = afd[afd_to_keep_idx]
            days_clean = days[afd_to_keep_idx]

            beta_estimate, sigma_estimate = simulation_utils.mle_sigma(afd_clean)

            #data = [days_clean, afd_clean]

            freq_value = 2*numpy.pi/365 # 0.01721420632
            freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
            freq_max = 2*numpy.pi/180 # 0.034906585 (365-185)

            phase_value = numpy.pi
            phase_min = 0
            phase_max = 2*numpy.pi

            param_mean_value = numpy.mean(afd)
            param_min_value = min(afd)
            param_max_value = max(afd)

            #print(param_mean_value)

            amp_value = 1
            amp_min = 0.001
            amp_max = 10

            beta_value = 1
            beta_min = 0.01
            beta_max = 1.95

            params = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                                        freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                        phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                        param_mean=dict(value=param_mean_value, min=param_min_value, max=param_max_value),
                                        beta=dict(value=beta_value, min=beta_min, max=beta_max))

            #  beta=dict(value=beta_value, min=beta_min, max=beta_max)

            #print(params)

            result_brute, fitter = grid_search_mle_sine_wave(days_clean, afd_clean, params)
            best_params_brute = result_brute.params

            print(best_params_brute)
            #for p in param_no_method_all:
            #    param_dict['%s_brute' % p].append(best_params_brute[p].value)

            #best_result_lbfgs = second_rount_optimization(result_brute, fitter, beta_estimate)

        
            #best_params_lbfgs = best_result_lbfgs.params
            #for p in param_no_method_all:
            #    param_dict['%s_mle' % p].append(best_params_lbfgs[p].value)

            #param_dict['beta'].append(beta_estimate)
            #param_dict['sigma'].append(sigma_estimate)


    sys.stderr.write("Saving parameter dictionary...\n")
    with open(param_dict_path, 'wb') as outfile:
        pickle.dump(param_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")




test_mle()
#make_plot()
