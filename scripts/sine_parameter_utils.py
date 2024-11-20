import config
import sys
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats
from scipy.optimize import leastsq, curve_fit, minimize
from scipy.special import loggamma
import lmfit
from lmfit import Minimizer, create_params, fit_report, conf_interval
# numdifftools also installed
import simulation_utils

import pickle

numpy.random.seed(123456789)


param_otu_dict_path = config.data_directory + 'param_otu_%s%s%sdict.pickle'
param_otu_mle_dict_path = config.data_directory + 'param_otu_mle_dict.pickle'

param_env_dict_path = config.data_directory + 'param_env_dict.pickle'


#param_leastsq_all = ['amp_leastsq', 'freq_leastsq', 'phase_leastsq']
param_no_method_all = ['amp', 'freq', 'phase', 'param_mean']

env_variable_all = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph', 'air_temperature']
log10_status_label_dict = {True:'log10_', False:''}
clr_status_label_dict = {True:'clr_', False: ''}
param_label_dict = {'amp': 'Amplitude', 'freq': 'Oscillation timescale (days)', 'phase': 'Phase'}
param_label_no_days_dict = {'amp': 'Amplitude', 'freq': 'Oscillation timescale', 'phase': 'Phase'}

param_label_dict_latex = {'amp': r'$A$', 'freq': r'$\tau^{\mathrm{env}}$', 'phase': r'$\psi$'}




metadata_dict = utils.build_metadata_dict()

minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()
s_by_s, otu_labels, samples = utils.load_count_data()
s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)


# get days
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


def calculate_sine_wave(t, amp, freq, phase, param_mean):

    return amp*numpy.sin(freq*t - phase) + param_mean


def get_param_otu_dict_path(log10_status=True, clr_status=False, otu_to_remove=None):

    if otu_to_remove == None:
        otu_to_remove_label = ''
    else:
        otu_to_remove_label = 'no_%s_' % otu_to_remove 

    log10_status_label = log10_status_label_dict[log10_status]
    clr_status_label = clr_status_label_dict[clr_status]

    param_dict_path_ = param_otu_dict_path % (log10_status_label, clr_status_label, otu_to_remove_label)

    return param_dict_path_



def fcn2min_sine(params, days_afd, afd):
   
    """Model sine wave, subtract data."""
    
    amp = params['amp']
    freq = params['freq']
    phase = params['phase']
    param_mean = params['param_mean']
    
    model = amp*(numpy.cos(phase)*numpy.sin(freq*days_afd) + numpy.sin(phase)*numpy.cos(freq*days_afd)) + param_mean
    #model = amp*numpy.sin(freq*days_afd - phase) + param_mean

    return model - afd



def fcn2min_sine_clipped(params, days_afd, afd, upper_bound=1):

    amp = params['amp']
    freq = params['freq']
    phase = params['phase']
    param_mean = params['param_mean']

    model = amp*(numpy.cos(phase)*numpy.sin(freq*days_afd) + numpy.sin(phase)*numpy.cos(freq*days_afd)) + param_mean
    
    # reset predictions to upper bound
    model[model>upper_bound] = upper_bound

    return model - afd


def fcn2min_sine_saturated(params, days_afd, afd, saturating_value=0.05):

    amp = params['amp']
    freq = params['freq']
    phase = params['phase']
    param_mean = params['param_mean']

    model = amp*(numpy.cos(phase)*numpy.sin(freq*days_afd) + numpy.sin(phase)*numpy.cos(freq*days_afd)) + param_mean
    
    # reset predictions to upper bound
    model = model/(saturating_value+model)

    return model - afd
    

def grid_search_sine_wave(days_afd_, afd_, params_, upper_bound=False):

    #upper_bound = True/False
    # saturating effect do to sampling...

    # brute force parameter search
    
    # default is leastsq
    # , method='leastsq'

    if upper_bound == True:
        #fitter = Minimizer(fcn2min_sine_clipped, params_, fcn_args=(days_afd_, afd_, upper_bound))
        fitter = Minimizer(fcn2min_sine_clipped, params_, fcn_args=(days_afd_, afd_, upper_bound))

    else:
        fitter = Minimizer(fcn2min_sine, params_, fcn_args=(days_afd_, afd_))

    # NS = number of grid points along the axes
    # keep = number of best candidates from the brute force method that are stored in the candidates attribute
    
    # brute force results
    result_brute = fitter.minimize(method='brute', Ns=30, keep=25)

    return result_brute, fitter



def fit_sine_wave_leastsq(days_afd, afd):

    guess_mean = numpy.mean(afd)
    # mean should be 1 because we rescaled
    #guess_std = 3*numpy.std(afd)#/(2**0.5)/(2**0.5)
    guess_phase = 30 # a month delay
    guess_freq = numpy.pi/365
    guess_amp = 2

    #model_first_guess = guess_std*numpy.sin(days+guess_phase) + guess_mean
    #model_first_guess = guess_std*numpy.sin(days+guess_phase)

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    #optimize_func = lambda x: x[0]*numpy.sin(x[1]*days+x[2]) + x[3] - afd

    # expand sin so that you only have one nonlinear parameter 
    optimize_func = lambda x: x[0]*(numpy.cos(x[2])*numpy.sin(x[1]*days_afd) + numpy.sin(x[2])*numpy.cos(x[1]*days_afd)) + x[3] - afd
    
    #optimize_func = lambda x: x[0]*numpy.sin(x[1]*days+x[2]) + 1 - afd
    #optimize_func = lambda x: x[0]*numpy.sin(x[1]*days+x[2]) - afd

    #est_amp, est_freq, est_phase = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase])[0]
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    return est_amp, est_freq, est_phase, est_mean



def second_rount_optimization(result_brute, fitter):

    # second round of optimization using least-squares with brute force as a starting point
    best_result_leastsq = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:
        trial = fitter.minimize(method='leastsq', params=candidate.params)
        if trial.chisqr < best_result_leastsq.chisqr:
            best_result_leastsq = trial
            #best_result_candidate = candidate

    return best_result_leastsq



def neg_ll_gamma_time_varying(params, days_afd, afd):

    amp = params['amp']
    freq = params['freq']
    phase = params['phase']
    param_mean = params['param_mean']

    # freq = 2*pi/tau





def ll_sine_gamma(params, days_afd, afd, beta):

    # minimize the negative log-likelihood 
    amp = params['amp']
    freq = params['freq']
    phase = params['phase']
    param_mean = params['param_mean']
    #beta = params['beta']

    x_bar_pred = numpy.exp(amp*(numpy.cos(phase)*numpy.sin(freq*days_afd) + numpy.sin(phase)*numpy.cos(freq*days_afd))) * param_mean
    
    ll = (beta-1)*sum(numpy.log(afd)) - beta*sum(afd/x_bar_pred) - beta*sum(numpy.log(x_bar_pred)) + len(afd)*beta*numpy.log(beta) - len(afd)*loggamma(beta)

    return -1*ll


def second_round_optimization_mle(result_brute, fitter, beta):

    # second round of optimization using least-squares with brute force as a starting point
    best_result_leastsq = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:
        trial = fitter.minimize(method='lbfgsb', params=candidate.params)
        if trial.chisqr < best_result_leastsq.chisqr:
            best_result_leastsq = trial

    return best_result_leastsq


def grid_search_mle_sine_wave(days_afd_, afd_, params_, beta_estimate):

    # minimize the negative log-likelihood 
    fitter = Minimizer(ll_sine_gamma, params_, fcn_args=(days_afd_, afd_, beta_estimate))
    
    # brute force results
    result_brute = fitter.minimize(method='brute', Ns=30, keep=25)

    return result_brute, fitter





def make_param_env_dict():

    param_dict = {}
    #param_types = ['amp_brute', 'amp_leastsq', 'freq_brute','freq_leastsq', 'phase_brute', 'phase_leastsq', 'param_mean_brute', 'param_mean_leastsq', 'upper_bound']
    for p in param_no_method_all:

        # list for environmental variables because they only have one data_type
        param_dict['%s_brute' % p] = []
        param_dict['%s_leastsq' % p] = []
        

    param_dict['env_variables_labels'] = env_variable_all

    sys.stderr.write("Fitting sine function to environmental variables...\n")
    sys.stderr.write(", ".join(["Env. variable", "Amplititude", "Frequency", "Phase", "Mean"]) + "\n")

    # environmental analysis....
    for env_variable_idx, env_variable in enumerate(env_variable_all):
        
        env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])
        # remove nans
        env_to_keep_idx = (~numpy.isnan(env_variable_array))
        env_variable_array_clean = env_variable_array[env_to_keep_idx]
        days_clean = days[env_to_keep_idx]
        
        #upper_bound = None

        freq_value = 2*numpy.pi/365 # 0.01721420632
        freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
        freq_max = 2*numpy.pi/180 # 0.034906585 (365-185)

        phase_value = numpy.pi
        phase_min = 0
        phase_max = 2*numpy.pi

        param_mean_value = numpy.mean(env_variable_array)
        param_min_value = min(env_variable_array)
        param_max_value = max(env_variable_array)

        amp_value = 1
        amp_min = 0.01
        amp_max = 10

        params = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                                freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                param_mean=dict(value=param_mean_value, min=param_min_value, max=param_max_value))


        result_brute, fitter = grid_search_sine_wave(days_clean, env_variable_array_clean, params)

        best_params_brute = result_brute.params
        for p in param_no_method_all:
            param_dict['%s_brute' % p].append(best_params_brute[p].value)


        #amp_brute = best_params_brute['amp'].value
        #freq_brute = best_params_brute['freq'].value
        #phase_brute = best_params_brute['phase'].value
        #param_mean_brute = best_params_brute['param_mean'].value
        
        best_result_leastsq = second_rount_optimization(result_brute, fitter)
        best_params_leastsq = best_result_leastsq.params
        for p in param_no_method_all:
            param_dict['%s_leastsq' % p].append(best_params_leastsq[p].value)
        
        sys.stderr.write("%s, %.4f, %.4f, %.4f, %.4f\n" % (env_variable, param_dict['amp_leastsq'][env_variable_idx], param_dict['freq_leastsq'][env_variable_idx], param_dict['phase_leastsq'][env_variable_idx], param_dict['param_mean_leastsq'][env_variable_idx]))


    sys.stderr.write("Saving parameter dictionary...\n")
    with open(param_env_dict_path, 'wb') as outfile:
        pickle.dump(param_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")




def make_param_otu_dict(log10_status=True, otu_to_remove=None, min_occupancy=1, clr_status=False):

    s_by_s, otu_labels, samples = utils.load_count_data()

    # remove the OTU

    if otu_to_remove != None:
        otu_to_keep_idx = (otu_labels != otu_to_remove)
        #otu_to_keep_idx = (otu_labels_subset != otu_to_remove)
        #rel_s_by_s_dna = rel_s_by_s_dna[otu_to_keep_idx,:]
        #rel_s_by_s_rna = rel_s_by_s_rna[otu_to_keep_idx,:]
        s_by_s = s_by_s[otu_to_keep_idx,:]
        otu_labels = otu_labels[otu_to_keep_idx]


    if clr_status == True:
        rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples, min_occupancy=1)
        #data_type_all = ['DNA', 'RNA']
        # rescale by subtracting the mean since it's already log transformed
        #rel_s_by_s_dna_rescaled = (rel_s_by_s_dna.T - numpy.mean(rel_s_by_s_dna, axis=1)).T
        #rel_s_by_s_rna_rescaled = (rel_s_by_s_rna.T - numpy.mean(rel_s_by_s_rna, axis=1)).T
        
        #rel_s_by_s_ratio = rel_s_by_s_rna - rel_s_by_s_dna
        #rel_s_by_s_ratio_rescaled = (rel_s_by_s_ratio.T - numpy.mean(rel_s_by_s_ratio, axis=1)).T


    else:
        # filter out otu
        rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=min_occupancy)
        # returns rescaled relative abundance
        # rescale relative abundances by dividing by the mean
        #rel_s_by_s_dna_rescaled = utils.rescale_s_by_s(rel_s_by_s_dna)
        #rel_s_by_s_rna_rescaled = utils.rescale_s_by_s(rel_s_by_s_rna)
        #rel_s_by_s_ratio = rel_s_by_s_rna/rel_s_by_s_dna
        #rel_s_by_s_ratio_rescaled = utils.rescale_s_by_s(rel_s_by_s_ratio)


    param_dict = {}
    param_dict['data'] = {}
    param_dict['data']['days'] = {}
    param_dict['data']['afd'] = {}
    #param_types = ['amp_brute', 'amp_leastsq', 'freq_brute','freq_leastsq', 'phase_brute', 'phase_leastsq', 'param_mean_brute', 'param_mean_leastsq', 'upper_bound']
    for p in param_no_method_all:
        
        # dictionary for OTUs because each OTU has multiple data types (RNA, DNA, ratio)
        param_dict['%s_brute' % p] = {}
        param_dict['%s_leastsq' % p] = {}

        param_dict['%s_leastsq_lower_ci' % p] = {}
        param_dict['%s_leastsq_upper_ci' % p] = {}


    param_dict['upper_bound'] = {}
    param_dict['otu_labels'] = otu_labels_subset.tolist()

    sys.stderr.write("Fitting sine wave to OTU timeseries...\n")
    sys.stderr.write(", ".join(["OTU", 'Sample type', "Amplititude", "Frequency", "Phase", "Mean"]) + "\n")
    for otu_idx in range(rel_s_by_s_dna.shape[0]):

        #if otu_labels_subset[otu_idx] != 'Otu000050':
        #    continue

        for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

            if data_type == 'DNA':
                #afd = rel_s_by_s_dna_rescaled[otu_idx,:]
                afd = rel_s_by_s_dna[otu_idx,:]

                print(afd)
                

            elif data_type == 'RNA':
                #afd = rel_s_by_s_rna_rescaled[otu_idx,:]
                afd = rel_s_by_s_rna[otu_idx,:]

            #else:
            #    afd = rel_s_by_s_rescaled_ratio[otu_idx,:]
           
            days_afd = numpy.copy(days)

            # only log transform if not using CLR
            if clr_status == False:

                if log10_status == True:
                    afd = numpy.log10(afd)

                else:
                    # remove rare outliers
                    to_keep_idx = (afd<=4)
                    afd = afd[to_keep_idx]
                    days_afd = days[to_keep_idx]


            # if a parameter does not have finite bounds, then it does need a brute_step attribute specified:
            freq_value = 2*numpy.pi/365 # 0.01721420632
            #freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
            freq_min = 2*numpy.pi/3000 # 0.01142397328 (365+185)
            #freq_max = 2*numpy.pi/180 # 0.034906585 (365-185)
            freq_max = 2*numpy.pi/30 # 0.20943951023 

            phase_value = numpy.pi
            phase_min = 0
            phase_max = 2*numpy.pi

            param_mean_value = numpy.mean(afd)

            if clr_status == True:
                amp_value = 1
                amp_min = 1e-3
                amp_max = 40

                #param_mean_min = -2
                #param_mean_max = 2

                print(numpy.mean(afd))

                param_mean_min = numpy.mean(afd) - 1
                param_mean_max = numpy.mean(afd) + 1


            else:
                
                if log10_status == True:
                    amp_value = 1

                    amp_min = 1e-3
                    amp_max = 3

                    #param_mean_min = numpy.log10(0.7) # -0.15490195998
                    #param_mean_max = numpy.log10(3) # 0.47712125472

                    param_mean_min = -0.5
                    param_mean_max = 0.5


                else:
                    amp_value = 1
                    amp_min = 1e-3
                    amp_max = 5

                    param_mean_min = -1
                    param_mean_max = 0
                

            params = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                                freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                param_mean=dict(value=param_mean_value, min=param_mean_min, max=param_mean_max))


            upper_bound_dict_otu_1 = {'RNA':0.2, 'DNA':0.8}
            #if (otu_labels_subset[otu_idx] == 'Otu000001') and ((data_type == 'RNA') or (data_type== 'DNA')): 
            if (otu_labels_subset[otu_idx] == 'Otu000001') and (data_type == 'RNA') and (clr_status==False):               
                upper_bound = upper_bound_dict_otu_1[data_type]
                result_brute, fitter = grid_search_sine_wave(days_afd, afd, params, upper_bound=upper_bound)
                
            else:
                upper_bound = None
                result_brute, fitter = grid_search_sine_wave(days_afd, afd, params)

            # initialize entries for this data type.
            if data_type not in param_dict['amp_brute']:
                
                param_dict['upper_bound'][data_type] = []
                param_dict['data']['days'][data_type] = []
                param_dict['data']['afd'][data_type] = []
                for p in param_no_method_all:
                    param_dict['%s_brute'% p][data_type] = []
                    param_dict['%s_leastsq'% p][data_type] = []

                    param_dict['%s_leastsq_lower_ci' % p][data_type] = []
                    param_dict['%s_leastsq_upper_ci' % p][data_type] = []


            # best parameters from brute force.
            best_params_brute = result_brute.params

            #def sine_resid(best_params_):
            #    return best_params_['amp']*(numpy.cos(best_params_['phase'])*numpy.sin(best_params_['freq']*days_afd) + numpy.sin(best_params_['phase'])*numpy.cos(best_params_['freq']*days_afd)) + best_params_['param_mean'] - afd

            for p in param_no_method_all:
                param_dict['%s_brute' % p][data_type].append(best_params_brute[p].value)

            
            best_result_leastsq = second_rount_optimization(result_brute, fitter)
            best_params_leastsq = best_result_leastsq.params
            ci = conf_interval(fitter, best_result_leastsq, sigmas=[0.95])
            
            for p in param_no_method_all:
                param_dict['%s_leastsq' % p][data_type].append(best_params_leastsq[p].value)
                param_dict['%s_leastsq_lower_ci' % p][data_type].append(ci[p][0][1])
                param_dict['%s_leastsq_upper_ci' % p][data_type].append(ci[p][2][1])


            param_dict['upper_bound'][data_type].append(upper_bound)
            # add AFD and days
            param_dict['data']['days'][data_type].append(days_afd.tolist())
            param_dict['data']['afd'][data_type].append(afd.tolist())

            sys.stderr.write("%s, %s, %.4f, %.4f, %.4f, %.4f\n" % (otu_labels_subset[otu_idx], data_type, param_dict['amp_leastsq'][data_type][otu_idx], param_dict['freq_leastsq'][data_type][otu_idx], param_dict['phase_leastsq'][data_type][otu_idx], param_dict['param_mean_leastsq'][data_type][otu_idx]))
            #sys.stderr.write("%s, %s, %.4f, %.4f, %.4f, %.4f\n" % (otu_labels_subset[otu_idx], data_type, param_dict['amp_leastsq'][data_type][0], param_dict['freq_leastsq'][data_type][0], param_dict['phase_leastsq'][data_type][0], param_dict['param_mean_leastsq'][data_type][0]))

            
    param_dict_path_ = get_param_otu_dict_path(log10_status, clr_status, otu_to_remove)
    sys.stderr.write("Saving parameter dictionary...\n")

    with open(param_dict_path_, 'wb') as outfile:
        pickle.dump(param_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stderr.write("Done!\n")




def make_param_mle_otu_dict(min_occupancy=1):

    s_by_s, otu_labels, samples = utils.load_count_data()
    rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples, min_occupancy=1)

    param_dict = {}
    param_dict['data'] = {}
    param_dict['data']['days'] = {}
    param_dict['data']['clr_afd'] = {}
    param_dict['beta'] = {}
    param_dict['sigma'] = {}
    for p in param_no_method_all:
        
        # dictionary for OTUs because each OTU has multiple data types (RNA, DNA, ratio)
        param_dict['%s_brute' % p] = {}
        param_dict['%s_mle' % p] = {}

        #param_dict['%s_mle_lower_ci' % p] = {}
        #param_dict['%s_mle_upper_ci' % p] = {}


    param_dict['otu_labels'] = otu_labels_subset.tolist()
    sys.stderr.write("Fitting sine wave to OTU timeseries...\n")
    sys.stderr.write(", ".join(["OTU", 'Sample type', "Amplititude", "Frequency", "Phase", "Mean"]) + "\n")
    for otu_idx in range(rel_s_by_s_dna.shape[0]):

        for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

            if data_type == 'DNA':
                clr_afd = rel_s_by_s_dna[otu_idx,:]                

            elif data_type == 'RNA':
                clr_afd = rel_s_by_s_rna[otu_idx,:]

            days_afd = numpy.copy(days)
            
            afd_to_keep_idx = (~numpy.isnan(clr_afd))
            clr_afd_clean = clr_afd[afd_to_keep_idx]
            days_afd_clean = days_afd[afd_to_keep_idx]
            afd_clean = numpy.exp(clr_afd_clean)

            freq_value = 2*numpy.pi/365 # 0.01721420632
            freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
            freq_max = 2*numpy.pi/180 # 0.034906585 (365-185)

            phase_value = numpy.pi
            phase_min = 0
            phase_max = 2*numpy.pi

            param_mean_value = numpy.mean(afd_clean)
            param_min_value = min(afd_clean)
            param_max_value = max(afd_clean)

            amp_value = 1
            amp_min = 0.001
            amp_max = 10


            params = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                                        freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                        phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                        param_mean=dict(value=param_mean_value, min=param_min_value, max=param_max_value))

            if data_type not in param_dict['amp_brute']:
                
                param_dict['data']['days'][data_type] = []
                param_dict['data']['clr_afd'][data_type] = []
                for p in param_no_method_all:
                    param_dict['%s_brute'% p][data_type] = []
                    param_dict['%s_mle'% p][data_type] = []

                    #param_dict['%s_mle_lower_ci' % p][data_type] = []
                    #param_dict['%s_mle_upper_ci' % p][data_type] = []

                param_dict['beta'][data_type] = []
                param_dict['sigma'][data_type] = []



            # get beta estimate
            beta_estimate, sigma_estimate = simulation_utils.mle_sigma(afd_clean)
            result_brute, fitter = grid_search_mle_sine_wave(days_afd_clean, afd_clean, params, beta_estimate)
            best_params_brute = result_brute.params

            param_dict['beta'][data_type].append(beta_estimate)
            param_dict['sigma'][data_type].append(sigma_estimate)

            for p in param_no_method_all:
                param_dict['%s_brute' % p][data_type].append(best_params_brute[p].value)

            
            best_result_mle = second_round_optimization_mle(result_brute, fitter, beta_estimate)
            best_params_mle = best_result_mle.params
            #ci = conf_interval(fitter, best_result_mle, sigmas=[0.95])

            for p in param_no_method_all:
                param_dict['%s_mle' % p][data_type].append(best_params_mle[p].value)
                #param_dict['%s_mle_lower_ci' % p][data_type].append(ci[p][0][1])
                #param_dict['%s_mle_upper_ci' % p][data_type].append(ci[p][2][1])


            param_dict['data']['days'][data_type].append(days_afd.tolist())
            param_dict['data']['clr_afd'][data_type].append(clr_afd_clean.tolist())

            sys.stderr.write("%s, %s, %.4f, %.4f, %.4f, %.4f\n" % (otu_labels_subset[otu_idx], data_type, param_dict['amp_mle'][data_type][otu_idx], param_dict['freq_mle'][data_type][otu_idx], param_dict['phase_mle'][data_type][otu_idx], param_dict['param_mean_mle'][data_type][otu_idx]))



    sys.stderr.write("Saving parameter dictionary...\n")
    with open(param_otu_mle_dict_path, 'wb') as outfile:
        pickle.dump(param_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")





def load_param_otu_dict(log10_status=False, clr_status=False, otu_to_remove=None):

    param_dict_path_ = get_param_otu_dict_path(log10_status, clr_status, otu_to_remove)

    dict_ = pickle.load(open(param_dict_path_, "rb"))
    return dict_


def load_param_env_dict():

    dict_ = pickle.load(open(param_env_dict_path, "rb"))
    return dict_




def plot_otu_1(data_type, log10_status=True, otu_to_remove=False, method='leastsq'):

    otu_idx = 0
    otu_label = 'Otu000001'

    param_dict = load_param_otu_dict(log10_status=log10_status, otu_to_remove=otu_to_remove)

    afd = s_by_s_rescaled_rna[otu_idx,:]

    fig, ax = plt.subplots(figsize=(4,4))

    amp = param_dict['amp_%s' % method][data_type][otu_idx]
    freq = param_dict['freq_%s' % method][data_type][otu_idx]
    phase = param_dict['phase_%s' % method][data_type][otu_idx]
    param_mean = param_dict['param_mean_%s' % method][data_type][otu_idx]

    upper_bound = 0.3

    ax.scatter(days, afd, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type])

    days_range = numpy.linspace(min(days), max(days), 1000)
    model_prediction = amp*numpy.sin(freq*days_range+phase)+param_mean
    model_prediction[model_prediction>upper_bound] = upper_bound
    model_prediction = 10**model_prediction

    ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict[data_type])
    
    ax.set_xlabel("Time (days)", fontsize=10)
    ax.set_ylabel(utils.rescaled_label_dict[data_type], fontsize=10)
    ax.set_title(otu_label, fontsize=11)
    ax.set_yscale('log', basey=10)

    # tick labels

    #minor_days, major_days, major_labels
    ax.set_xlim([0, max(days)])
    ax.set_xticks(minor_days, minor=True)
    ax.set_xticks(major_days, minor=False)
    ax.set_xticklabels(major_labels, minor=False, fontsize=7)

    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%sotu_1_%s.png" % (config.analysis_directory, data_type)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def plot_fits(data_type='ratio', log10_status=True, otu_to_remove=None, method='leastsq'):

    param_dict = load_param_otu_dict(log10_status=log10_status, otu_to_remove=otu_to_remove)

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(param_dict['otu_labels'])))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))
            days = numpy.asarray(param_dict['data']['days'][data_type][c])
            afd = numpy.asarray(param_dict['data']['afd'][data_type][c])

            amp = param_dict['amp_%s' % method][data_type][c]
            freq = param_dict['freq_%s' % method][data_type][c]
            phase = param_dict['phase_%s' % method][data_type][c]
            param_mean = param_dict['param_mean_%s' % method][data_type][c]
            upper_bound = param_dict['upper_bound'][data_type][c]

            days_range = numpy.linspace(min(days), max(days), 1000)
            model_prediction = amp*numpy.sin(freq*days_range+phase)+param_mean

            if upper_bound != None:
                upper_bound = float(upper_bound)
                model_prediction[model_prediction>upper_bound] = upper_bound

            if log10_status == True:
                afd = 10**afd
                model_prediction = 10**model_prediction

            ax.scatter(days, afd, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type])
            ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict[data_type])
            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel(utils.rescaled_label_dict[data_type], fontsize=10)
            ax.set_title(param_dict['otu_labels'][c], fontsize=11)

            #minor_days, major_days, major_labels
            ax.set_xlim([0, max(days)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)

            ax.set_ylim([min(afd), max(afd)])

            if log10_status == True:
                ax.set_yscale('log', basey=10)


    if otu_to_remove == None:
        otu_to_remove_label = ''
    else:
        otu_to_remove_label = '_no_%s' % otu_to_remove 
 
    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%ssine_fits_%s%s.png" % (config.analysis_directory, data_type, otu_to_remove_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_params(log10_status=False, clr_status=False, method='leastsq'):

    param_dict = load_param_otu_dict(log10_status=log10_status, clr_status=clr_status,)

    fig = plt.figure(figsize = (12, 8))
    fig.subplots_adjust(bottom= 0.15)

    for param_idx, param in enumerate(param_no_method_all):
        
        # skip parameter mean
        if param == 'param_mean':
            continue

        param_label  = '%s_%s' % (param, method)
        
        ax_dist = plt.subplot2grid((2, 3), (0, param_idx), colspan=1)
        ax_compare = plt.subplot2grid((2, 3), (1, param_idx), colspan=1)

        # ratio as reference
        param_reference_idx = numpy.argsort(param_dict[param_label]['DNA'])

        if param_label == 'amp_leastsq':
            ax_dist.axvline(x=0, lw=2, ls=':', c='k', label='No oscillations')

        elif param_label == 'freq_leastsq':
            #ax_dist.axvline(x=0, lw=2, ls=':', c='k', label='Freq. ' + r'$=0$')
            ax_dist.axvline(x=2*numpy.pi/365, lw=2, ls='--', c='k', label='Yearly oscillations')
            ax_dist.axvline(x=2*numpy.pi/91.25, lw=2, ls=':', c='k', label='Seasonal oscillations')

        else:
            ax_dist.axvline(x=0, lw=2, ls=':', c='k')


        #for t_idx, t in enumerate(utils.data_type_all):
        for t_idx, t in enumerate(['DNA', 'RNA']):

            param_t = param_dict[param_label][t]

            ax_dist.hist(param_t, 8, histtype='step', density=True, stacked=True, fill=False, color=utils.dna_rna_color_dict[t], label=utils.rescaled_label_clr_dict[t])
            ax_dist.set_xlabel(param_label_dict[param_label], fontsize=11)
        
            param_t = numpy.asarray(param_t)
            #ax_compare.scatter(param_t[param_reference_idx], list(range(len(param_t))), s=6, color=utils.dna_rna_color_dict[t], zorder=2)
            #ax_compare.plot(param_t, list(range(len(param_t))), lw=1, alpha=0.6, color=utils.dna_rna_color_dict[t], zorder=1)

        ax_dist.set_xlabel(param_label_dict[param_label], fontsize=10)
        ax_dist.set_ylabel("Probability density", fontsize=10)


        param_dna = numpy.asarray(param_dict[param_label]['DNA'])
        param_rna = numpy.asarray(param_dict[param_label]['RNA'])

        param_merged = numpy.concatenate([param_dna, param_rna])

        param_dna_sorted = param_dna[param_reference_idx]
        param_rna_sorted = param_rna[param_reference_idx]
        rho_param = numpy.corrcoef(param_rna_sorted, param_dna_sorted)[0,1]

        min_param_merged = min(param_merged)/1.2
        max_param_merged = max(param_merged)*1.2

        ax_compare.scatter(param_dna_sorted, param_rna_sorted, s=6, color='k', alpha=0.6, zorder=2)
        ax_compare.plot([min_param_merged, max_param_merged], [min_param_merged, max_param_merged], lw=1, ls=':', alpha=1, color='k', zorder=1, label='1:1')

        ax_compare.set_xlim([min_param_merged, max_param_merged])
        ax_compare.set_ylim([min_param_merged, max_param_merged])
        #ax_compare.set_xlabel(param_label_dict[param_label], fontsize=10)
        #ax_compare.set_yticks(list(range(len(param_t))))
        #ax_compare.set_yticklabels(otu_labels_subset[param_reference_idx], fontsize=6)

        ax_compare.text(0.8, 0.24, r'$\rho^{2} = $' + str(round(rho_param**2, 3)), fontsize=10, ha='center', va='center', transform=ax_compare.transAxes)

        ax_compare.set_xlabel('%s, DNA' % param_label_dict[param_label], fontsize=11)
        ax_compare.set_ylabel('%s, RNA' % param_label_dict[param_label], fontsize=11)


        if param_idx == 0:
            ax_dist.legend(loc='upper right', fontsize=6)

        if param_idx == 1:
            ax_dist.legend(loc='upper right', fontsize=6)

        # recreate the fitted curve using the optimized parameters
        #model_fit = est_amp*numpy.sin(est_freq*days+est_phase) + est_mean
        #model_fit = est_amp*numpy.sin(est_freq*days+est_phase)


    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    fig_name = "%ssine_parameters.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_data_collapse_timeseris(log10_status=True, method='leastsq', env_variable='water_temp'):

    metadata_dict = utils.build_metadata_dict()

    data_label = [r'$\tilde{x}_{i}^{(d)}(t)$', r'$\tilde{x}_{i}^{(r)}(t)$', r'$\phi_{i}(t)$']
    data_collapse_label = [r'$A^{-1}\left (  \tilde{x}_{i}^{(d)}(t) - \left< \tilde{x}_{i}^{(d)} \right> \right )$', r'$A^{-1}\left (  \tilde{x}_{i}^{(r)}(t) - \left<   \tilde{x}_{i}^{(d)} \right> \right )$', r'$A^{-1}\left (  \phi_{i}(t) - \left<   \tilde{x}_{i}^{(d)} \right> \right )$']

    param_dict = load_param_otu_dict(log10_status=log10_status)

    fig = plt.figure(figsize = (20, 18))
    fig.subplots_adjust(bottom= 0.15)


    # environmental variable
    #env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])
    # remove nans
    #env_to_keep_idx = (~numpy.isnan(env_variable_array))
    #env_variable_array = env_variable_array[env_to_keep_idx]
    #days_env = days[env_to_keep_idx]

    s_by_s, otu_labels, samples = utils.load_count_data()
    s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

    # returns rescaled relative abundance
    s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
    s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)
    s_by_s_rescaled_ratio = s_by_s_rescaled_rna/s_by_s_rescaled_dna


    #temp_idx = param_dict['env_variables']['env_variables_labels'].index(env_variable)

    #amp_temp = param_dict['env_variables']['amp_%s' % method][temp_idx]
    #freq_temp = param_dict['env_variables']['freq_%s' % method][temp_idx]
    #phase_temp = param_dict['env_variables']['phase_%s' % method][temp_idx]
    #param_mean_temp = param_dict['env_variables']['param_mean_%s' % method][temp_idx]
    
    #rescaled_env_variable_array = (env_variable_array - param_mean_temp)/amp_temp
    #rescaled_days_env = freq_temp*days_env + phase_temp

    for data_type_idx, data_type in enumerate(utils.data_type_all):
        
        phase_all = []

        ax_data = plt.subplot2grid((3, 3), (0, data_type_idx), colspan=1)
        ax_rescaled_data_y = plt.subplot2grid((3, 3), (1, data_type_idx), colspan=1)
        ax_rescaled_data_xy = plt.subplot2grid((3, 3), (2, data_type_idx), colspan=1)

        for otu_idx in range(s_by_s_rescaled_ratio.shape[0]):

            if data_type == 'DNA':
                afd = s_by_s_rescaled_dna[otu_idx,:]

            elif data_type == 'RNA':
                afd = s_by_s_rescaled_rna[otu_idx,:]

            else:
                afd = s_by_s_rescaled_ratio[otu_idx,:]


            if log10_status == True:
                afd = numpy.log10(afd)
                days_afd = numpy.copy(days)

            else:
                # remove rare outliers
                to_keep_idx = (afd<=4)
                afd = afd[to_keep_idx]
                days_afd = days[to_keep_idx]
            

            amp = param_dict['amp_%s' % method][data_type][otu_idx]
            freq = param_dict['freq_%s' % method][data_type][otu_idx]
            phase = param_dict['phase_%s' % method][data_type][otu_idx]
            param_mean = param_dict['param_mean_%s' % method][data_type][otu_idx]
            #upper_bound = param_dict['upper_bound'][data_type][otu_idx]

            phase_all.append(phase)

            ax_data.scatter(days_afd, afd, s=1, alpha=0.7, c=utils.dna_rna_color_dict[data_type], zorder=2)
            ax_data.plot(days_afd, afd, alpha=0.2, ls='-', lw=0.5, c=utils.dna_rna_color_dict[data_type], zorder=1)

            rescaled_days = freq*days_afd + phase
            afd_rescaled = (afd - param_mean)/amp

            ax_rescaled_data_y.scatter(days_afd, afd_rescaled, s=1, alpha=0.7, c=utils.dna_rna_color_dict[data_type], zorder=2)
            ax_rescaled_data_y.plot(days_afd, afd_rescaled, alpha=0.2, ls='-', lw=0.5, c=utils.dna_rna_color_dict[data_type], zorder=1)


            ax_rescaled_data_xy.scatter(rescaled_days, afd_rescaled, s=1, alpha=0.7, c=utils.dna_rna_color_dict[data_type], zorder=2)
            ax_rescaled_data_xy.plot(rescaled_days, afd_rescaled, alpha=0.2, ls='-', lw=0.5, c=utils.dna_rna_color_dict[data_type], zorder=1)


            #ax.set_xlabel("Time (days)", fontsize=10)
        # plot environmental variables
        #ax_rescaled_data_y.plot(days_env, rescaled_env_variable_array, alpha=1, ls='-', lw=3, c='k', zorder=4, label='Param. rescaled water temp.')
        #ax_rescaled_data_xy.plot(rescaled_days_env, rescaled_env_variable_array, alpha=1, ls='-', lw=3, c='k', zorder=4, label='Param. rescaled water temp.')

        
        phase_all = numpy.asarray(phase_all)

        #ax.set_xlabel("Time (days), " + r'$t$', fontsize=10)
        ax_data.set_title(utils.sample_label_dict[data_type], fontsize=12)
        ax_data.set_xlabel("Time (days), " + r'$t$', fontsize=12)
        ax_data.set_ylabel("Rescaled relative abundance, " + data_label[data_type_idx], fontsize=12)

        ax_rescaled_data_y.set_xlabel("Time (days), " + r'$t$', fontsize=12)
        ax_rescaled_data_y.set_ylabel("Parameter rescaled relative\nabundance, " + data_collapse_label[data_type_idx], fontsize=12)
        

        ax_rescaled_data_xy.set_xlabel("Parameter rescaled time, " + r'$ \frac{t}{\tau_{i, env}} + \psi_{i}$', fontsize=12)
        ax_rescaled_data_xy.set_ylabel("Parameter rescaled relative\nabundance, " + data_collapse_label[data_type_idx], fontsize=12)
        
        if data_type_idx == 0:
            ax_rescaled_data_y.legend(loc = 'lower right')
            ax_rescaled_data_xy.legend(loc = 'lower right')


        ax_data.set_xlim([0, max(days)])
        ax_data.set_xticks(minor_days, minor=True)
        ax_data.set_xticks(major_days, minor=False)
        ax_data.set_xticklabels(major_labels, minor=False, fontsize=7)

        ax_rescaled_data_y.set_xlim([0, max(days)])
        ax_rescaled_data_y.set_xticks(minor_days, minor=True)
        ax_rescaled_data_y.set_xticks(major_days, minor=False)
        ax_rescaled_data_y.set_xticklabels(major_labels, minor=False, fontsize=7)

        #ax_rescaled_data_xy.set_xlim([0, max(rescaled_days)])
        #ax_rescaled_data_xy.set_xticks(minor_days, minor=True)
        #ax_rescaled_data_xy.set_xticks(major_days, minor=False)
        #ax_rescaled_data_xy.set_xticklabels(major_labels, minor=False, fontsize=7)


        #if data_type_idx == 0:
        #    ax.legend(loc='lower left')


    fig.subplots_adjust(hspace=0.25, wspace=0.30)
    fig_name = "%sdata_collapse_timeseries.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_rna_dna_residuals():

    metadata_dict = utils.build_metadata_dict()

    s_by_s, otu_labels, samples = utils.load_count_data()
    s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

    # returns rescaled relative abundance
    s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
    s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)
    #s_by_s_rescaled_ratio = s_by_s_rescaled_rna/s_by_s_rescaled_dna

    # get days
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

    param_dict = load_param_otu_dict(log10_status=True)

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels_subset)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            afd_dna = s_by_s_rescaled_dna[c,:]
            afd_rna = s_by_s_rescaled_rna[c,:]

            afd_log10_dna = numpy.log10(afd_dna)
            afd_log10_rna = numpy.log10(afd_rna)

            afd_log10_dna_predicted = param_dict['amp_leastsq']['DNA'][c]*numpy.sin(param_dict['freq_leastsq']['DNA'][c]*days+param_dict['phase_leastsq']['DNA'][c])+param_dict['param_mean_leastsq']['DNA'][c]
            afd_log10_rna_predicted = param_dict['amp_leastsq']['RNA'][c]*numpy.sin(param_dict['freq_leastsq']['RNA'][c]*days+param_dict['phase_leastsq']['RNA'][c])+param_dict['param_mean_leastsq']['RNA'][c]

            resid_afd_log10_dna = afd_log10_dna - afd_log10_dna_predicted
            resid_afd_log10_rna = afd_log10_rna - afd_log10_rna_predicted

            ax.hist(resid_afd_log10_dna, bins=15, color=utils.dna_rna_color_dict['DNA'], alpha=0.7, density=True, zorder=2, label='DNA')
            ax.hist(resid_afd_log10_rna, bins=15, color=utils.dna_rna_color_dict['RNA'], alpha=0.7, density=True, zorder=1, label='RNA')


            ax.set_xlabel("Residuals of sine model", fontsize=10)
         
            ax.set_title(param_dict['otu_labels'][c], fontsize=11)

            if c == 0:
                ax.legend(loc = 'lower right')


    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%srna_dna_residuals.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def plot_compare_rna_dna_residuals():

    metadata_dict = utils.build_metadata_dict()

    #minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()

    s_by_s, otu_labels, samples = utils.load_count_data()
    s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

    # returns rescaled relative abundance
    s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
    s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)
    s_by_s_rescaled_ratio = s_by_s_rescaled_rna/s_by_s_rescaled_dna

    # get days
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

    param_dict = load_param_otu_dict(log10_status=True)


    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels_subset)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            afd_dna = s_by_s_rescaled_dna[c,:]
            afd_rna = s_by_s_rescaled_rna[c,:]

            afd_log10_dna = numpy.log10(afd_dna)
            afd_log10_rna = numpy.log10(afd_rna)

            afd_log10_dna_predicted = param_dict['amp_leastsq']['DNA'][c]*numpy.sin(param_dict['freq_leastsq']['DNA'][c]*days+param_dict['phase_leastsq']['DNA'][c])+param_dict['param_mean_leastsq']['DNA'][c]
            afd_log10_rna_predicted = param_dict['amp_leastsq']['RNA'][c]*numpy.sin(param_dict['freq_leastsq']['RNA'][c]*days+param_dict['phase_leastsq']['RNA'][c])+param_dict['param_mean_leastsq']['RNA'][c]

            resid_afd_log10_dna = afd_log10_dna - afd_log10_dna_predicted
            resid_afd_log10_rna = afd_log10_rna - afd_log10_rna_predicted

            ax.scatter(resid_afd_log10_dna, resid_afd_log10_rna, s=8, alpha=1, c='k', zorder=2)
            #ax.set_xlim([0, max(days)])
            #ax.set_xticks(minor_days, minor=True)
            #ax.set_xticks(major_days, minor=False)
            #ax.set_xticklabels(major_labels, minor=False, fontsize=7)

            ax.set_xlabel("Residuals of sine model, DNA", fontsize=10)
            ax.set_ylabel("Residuals of sine model, RNA", fontsize=10)

            merge_resid = numpy.concatenate((resid_afd_log10_dna, resid_afd_log10_rna), axis=None)

            min_resid = min(merge_resid)
            max_resid = max(merge_resid)

            ax.set_xlim([min_resid, max_resid])
            ax.set_ylim([min_resid, max_resid])

            ax.plot([min_resid, max_resid], [min_resid, max_resid], ls=':', lw=2, zorder=1, label='1:1')
            ax.set_title(param_dict['otu_labels'][c], fontsize=11)

            # correlation
            rho = numpy.corrcoef(resid_afd_log10_dna, resid_afd_log10_rna)[0,1]

            ax.text(0.24, 0.8, r'$\rho^{2} = $' + str(round(rho**2, 3)), fontsize=15, ha='center', va='center', transform=ax.transAxes)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(resid_afd_log10_dna, resid_afd_log10_rna)
            x_range_ =  numpy.linspace(min_resid*1.2, max_resid*0.8, 10000)
            y_fit_range = slope*x_range_ + intercept
            ax.plot(x_range_, y_fit_range, lw=2.5, ls='--', c='k', label='OLS regression', zorder=3)

            if c == 0:
                ax.legend(loc = 'lower right')


    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%scompare_rna_dna_residuals.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_sine_residuals_all():

    fig = plt.figure(figsize = (12, 4))
    fig.subplots_adjust(bottom= 0.15)

    metadata_dict = utils.build_metadata_dict()
    param_dict = load_param_otu_dict(log10_status=True)

    s_by_s, otu_labels, samples = utils.load_count_data()
    s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

    # returns rescaled relative abundance
    s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
    s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)
    s_by_s_rescaled_ratio = s_by_s_rescaled_rna/s_by_s_rescaled_dna

    s_by_s_rescaled_dna_log = numpy.log10(s_by_s_rescaled_dna)
    s_by_s_rescaled_rna_log = numpy.log10(s_by_s_rescaled_rna)
    s_by_s_rescaled_ratio_log = numpy.log10(s_by_s_rescaled_ratio)

    s_by_s_rescaled_log_dict = {'DNA':s_by_s_rescaled_dna_log, 'RNA':s_by_s_rescaled_rna_log, 'ratio':s_by_s_rescaled_ratio_log}

    for d_idx, d in enumerate(utils.data_type_all):

        ax = plt.subplot2grid((1, 3), (0, d_idx))

        s_by_s_rescaled = s_by_s_rescaled_log_dict[d]

        resid_all = []
        hist_resid_all = []
        for otu_i_idx in range(len(otu_labels_subset)):
            
            resid_i = s_by_s_rescaled[otu_i_idx,:] - (param_dict['amp_leastsq'][d][otu_i_idx]*numpy.sin(param_dict['freq_leastsq'][d][otu_i_idx]*days+param_dict['phase_leastsq'][d][otu_i_idx])+param_dict['param_mean_leastsq'][d][otu_i_idx])
            
            #rescaled_resid_i = (resid_i - numpy.mean(resid_i))/numpy.std(resid_i)
            hist_resid_i, bins_resid_i = utils.get_hist_and_bins(resid_i, bins=10)
            ax.scatter(bins_resid_i, hist_resid_i, s=7, color=utils.dna_rna_color_dict[d], alpha=0.7, lw=1)

            hist_resid_all.append(hist_resid_i)

            resid_all.append(resid_i)


        resid_all = numpy.concatenate(resid_all).ravel()
        hist_resid_all = numpy.concatenate(hist_resid_all).ravel()

        # fit loggamma
        shape_gamma, loc_gamma, scale_gamma = stats.loggamma.fit(resid_all)
        x = numpy.linspace(stats.loggamma.ppf(0.001, shape_gamma, loc=loc_gamma, scale=scale_gamma), stats.loggamma.ppf(0.999, shape_gamma, loc=loc_gamma, scale=scale_gamma), 100)
        pdf_loggamma_to_plot = stats.loggamma.pdf(x, shape_gamma, loc=loc_gamma, scale=scale_gamma)
        ax.plot(x, pdf_loggamma_to_plot, 'k', ls='--', lw=3, label='Gamma')

        ax.set_ylim([min(hist_resid_all), max(hist_resid_all)])

        print(shape_gamma * scale_gamma, shape_gamma)


        ax.set_yscale('log', basey=10)
        ax.set_title(utils.sample_label_dict[d], fontsize=12)
        ax.set_xlabel("Residuals of sine fit", fontsize = 10)
        ax.set_ylabel("Probability density", fontsize = 10)

        if d_idx == 0:
            ax.legend(loc='upper left', fontsize=10)
        
        #afd_log10_dna_predicted = 


    fig.subplots_adjust(hspace=0.35,wspace=0.25)
    fig_name = "%ssine_residuals.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()






def plot_time_vs_residuals(data_type='ratio', log10_status=True, otu_to_remove=None, method='leastsq'):

    param_dict = load_param_otu_dict(log10_status=log10_status, otu_to_remove=otu_to_remove)

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(param_dict['otu_labels'])))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))
            days = numpy.asarray(param_dict['data']['days'][data_type][c])
            afd = numpy.asarray(param_dict['data']['afd'][data_type][c])

            amp = param_dict['amp_%s' % method][data_type][c]
            freq = param_dict['freq_%s' % method][data_type][c]
            phase = param_dict['phase_%s' % method][data_type][c]
            param_mean = param_dict['param_mean_%s' % method][data_type][c]
            upper_bound = param_dict['upper_bound'][data_type][c]

            #days_range = numpy.linspace(min(days), max(days), 1000)
            model_prediction = amp*numpy.sin(freq*days+phase)+param_mean

            if upper_bound != None:
                upper_bound = float(upper_bound)
                model_prediction[model_prediction>upper_bound] = upper_bound

            #if log10_status == True:
            #    afd = 10**afd
            #    model_prediction = 10**model_prediction

            residuals = afd - model_prediction

            ax.scatter(days, residuals, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type], zorder=1)
            ax.axhline(y=0, ls=':', lw=2, zorder=0)#')
            #ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict[data_type])
            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel(utils.rescaled_label_dict[data_type] + ' residuals', fontsize=10)
            ax.set_title(param_dict['otu_labels'][c], fontsize=11)

            #minor_days, major_days, major_labels
            ax.set_xlim([0, max(days)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)
            max_ = numpy.absolute(max(residuals))
            ax.set_ylim([-1*max_, max_])

            #if log10_status == True:
            #    ax.set_yscale('log', basey=10)


    if otu_to_remove == None:
        otu_to_remove_label = ''
    else:
        otu_to_remove_label = '_no_%s' % otu_to_remove 
 
    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_residuals_%s%s.png" % (config.analysis_directory, data_type, otu_to_remove_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_rescaled_data_with_vs_without_otu1(data_type='RNA', log10_status=True):

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    param_dict = load_param_otu_dict(log10_status=log10_status)
    param_dict_no_otu1 = load_param_otu_dict(log10_status=log10_status, otu_to_remove='Otu000001')

    idx_all = list(range(len(param_dict_no_otu1['otu_labels'])))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))
            days = numpy.asarray(param_dict['data']['days'][data_type][c+1])
            afd = numpy.asarray(param_dict['data']['afd'][data_type][c+1])

            days_no_otu1 = numpy.asarray(param_dict_no_otu1['data']['days'][data_type][c])
            afd_no_otu1 = numpy.asarray(param_dict_no_otu1['data']['afd'][data_type][c])


            ax.set_xlabel("Time (days), " + r'$t$', fontsize=12)
            ax.set_ylabel("Rescaled relative abundance", fontsize=12)


            #ax.scatter(days, afd, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type], zorder=2, label='W/ OTU1')
            ax.scatter(days_no_otu1, afd_no_otu1, s=8, alpha=1, facecolors='none', edgecolors=utils.dna_rna_color_dict[data_type], zorder=2, label='W/out OTU1')
            #ax.plot(days, afd, lw=1, ls='-', c='k', zorder=1)
            ax.plot(days_no_otu1, afd_no_otu1, lw=1, ls=':', c='k', zorder=1)


            ax.set_title(param_dict_no_otu1['otu_labels'][c], fontsize=11)
            
            ax.axhline(y=0, ls=':', lw=2, zorder=0)#')


            ax.set_xlim([0, max(days)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)

            if c == 0:
                ax.legend(loc = 'lower right')



    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%swith_vs_without_otu1_%s.png" % (config.analysis_directory, data_type)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()






def plot_time_vs_abundance_clr(data_type='RNA', otu_to_remove=None, method='mle'):

    metadata_dict = utils.build_metadata_dict()
    s_by_s, otu_labels, samples = utils.load_count_data()
    
    if otu_to_remove != None:
        otu_to_keep_idx = (otu_labels != otu_to_remove)
        s_by_s = s_by_s[otu_to_keep_idx,:]
        otu_labels = otu_labels[otu_to_keep_idx]

    clr_s_by_s_dna, clr_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples)

    # get days
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type==data_type)]])


    #param_dict =  load_param_otu_dict(log10_status=False, clr_status=True)
    param_dict =  pickle.load(open(param_otu_mle_dict_path, 'rb'))

    days = param_dict['data']['days'][data_type]
    afd = param_dict['data']['clr_afd'][data_type]

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(afd)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            days_c = days[c]
            afd_c = afd[c]

            amp = param_dict['amp_%s' % method][data_type][c]
            freq = param_dict['freq_%s' % method][data_type][c]
            phase = param_dict['phase_%s' % method][data_type][c]
            param_mean = param_dict['param_mean_%s' % method][data_type][c]
            #upper_bound = param_dict['upper_bound'][data_type][c]
            beta = param_dict['beta'][data_type][c]

            days_range = numpy.linspace(min(days_c), max(days_c), 1000)
            model_prediction = amp*numpy.sin(freq*days_range+phase) + numpy.log(param_mean)

            #ax.scatter(days, afd, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type])
            #ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict[data_type], zorder=1)
            ax.plot(days_range, model_prediction, ls='-', lw=3, c=utils.dna_rna_color_dict[data_type], zorder=1)

            ax.scatter(days_c, afd_c, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type], zorder=2)
            #ax.axhline(y=0, ls=':', lw=2, zorder=0, c='k')#')
            #ax.plot(days_range, model_prediction, ls='-', lw=1, c=utils.dna_rna_color_dict[data_type])
            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel("CLR transformed abundance, " + utils.rescaled_label_clr_dict[data_type], fontsize=10)
            ax.set_title(otu_labels_subset[c], fontsize=11)

            #minor_days, major_days, major_labels
            ax.set_xlim([0, max(days_c)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)
            #max_ = numpy.absolute(max(residuals))
            #ax.set_ylim([-1*max_, max_])

            #if (chunk_idx == 0) and (c_idx == 0):
            #    ax.legend(loc='upper right', fontsize=6)


    if otu_to_remove == None:
        otu_to_remove_label = ''
    else:
        otu_to_remove_label = '_no_%s' % otu_to_remove 
 
    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_abundance_clr_%s%s.png" % (config.analysis_directory, data_type, otu_to_remove_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def plot_clr_abundance_with_vs_without_otu1(data_type='RNA'):
    

    metadata_dict = utils.build_metadata_dict()
    s_by_s, otu_labels, samples = utils.load_count_data()
    
    otu_to_keep_idx = (otu_labels != 'Otu000001')
    s_by_s_no_otu1 = s_by_s[otu_to_keep_idx]
    otu_labels_no_otu1 = otu_labels[otu_to_keep_idx]

    clr_s_by_s_dna, clr_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples)
    clr_s_by_s_no_otu1_dna, clr_s_by_s_no_otu1_rna, otu_labels_no_otu1_subset = utils.clr_transform(s_by_s_no_otu1, otu_labels_no_otu1, samples)

     # get days
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type==data_type)]])


    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(clr_s_by_s_no_otu1_dna.shape[0]))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]    

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            if data_type == 'DNA':
                clr_c = clr_s_by_s_dna[c+1,:]
                clr_no_otu1_c = clr_s_by_s_no_otu1_dna[c,:]

            else:
                clr_c = clr_s_by_s_rna[c+1,:]
                clr_no_otu1_c = clr_s_by_s_no_otu1_rna[c,:]

            # rescale by mean for comparison
            clr_c = clr_c - numpy.mean(clr_c)
            clr_no_otu1_c = clr_no_otu1_c - numpy.mean(clr_no_otu1_c)

            ax.scatter(days, clr_c, s=8, alpha=1, c=utils.dna_rna_color_dict[data_type], zorder=2, label='W/ OTU1')
            ax.plot(days, clr_c, lw=1, ls='-', c='k', zorder=1)
           
            ax.scatter(days, clr_no_otu1_c, s=8, alpha=1, facecolors='none', edgecolors=utils.dna_rna_color_dict[data_type], zorder=2, label='W/out OTU1')
            ax.plot(days, clr_no_otu1_c, lw=1, ls=':', c='k', zorder=1)


            ax.set_title(otu_labels_no_otu1_subset[c], fontsize=11)
            #ax.axhline(y=0, ls=':', lw=2, zorder=0)#')

            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel("Rescaled CLR transformed abundance", fontsize=8)

            #minor_days, major_days, major_labels
            ax.set_xlim([0, max(days)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)



    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_abundance_w_vs_wout_otu1_clr_%s.png" % (config.analysis_directory, data_type)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_time_vs_clr_ratio(method='mle'):

    metadata_dict = utils.build_metadata_dict()
    s_by_s, otu_labels, samples = utils.load_count_data()
    clr_s_by_s_dna, clr_s_by_s_rna, otu_labels_subset = utils.clr_transform(s_by_s, otu_labels, samples)

    # get days
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    #days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='DNA')]])


    #param_dict =  load_param_otu_dict(log10_status=False, clr_status=True)
    param_dict =  pickle.load(open(param_otu_mle_dict_path, 'rb'))

    days_dna = param_dict['data']['days']['DNA']
    afd_dna = param_dict['data']['clr_afd']['DNA']

    days_rna = param_dict['data']['days']['RNA']
    afd_rna = param_dict['data']['clr_afd']['RNA']

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(afd_rna)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            days_dna_c = numpy.asarray(days_dna[c])
            days_rna_c = numpy.asarray(days_rna[c])

            afd_dna_c = numpy.asarray(afd_dna[c])
            afd_rna_c = numpy.asarray(afd_rna[c])

            days_intersect_c = numpy.intersect1d(days_dna_c, days_rna_c)

            to_keep_dna_idx = numpy.asarray([numpy.where(days_dna_c==d)[0][0] for d in days_intersect_c])
            to_keep_rna_idx = numpy.asarray([numpy.where(days_rna_c==d)[0][0] for d in days_intersect_c])

            days_dna_c = days_dna_c[to_keep_dna_idx]
            days_rna_c = days_rna_c[to_keep_rna_idx]

            afd_dna_c = afd_dna_c[to_keep_dna_idx]
            afd_rna_c = afd_rna_c[to_keep_dna_idx]

            amp_dna = param_dict['amp_%s' % method]['DNA'][c]
            amp_rna = param_dict['amp_%s' % method]['RNA'][c]

            freq_dna = param_dict['freq_%s' % method]['DNA'][c]
            freq_rna = param_dict['freq_%s' % method]['RNA'][c]

            phase_dna = param_dict['phase_%s' % method]['DNA'][c]
            phase_rna = param_dict['phase_%s' % method]['RNA'][c]
            
            param_mean_dna = param_dict['param_mean_%s' % method]['DNA'][c]
            param_mean_rna = param_dict['param_mean_%s' % method]['RNA'][c]

            # rescale AFD for sine calculation
            rescaled_afd_dna_c = (afd_dna_c - numpy.log(param_mean_dna))/amp_dna
            rescaled_afd_rna_c = (afd_rna_c - numpy.log(param_mean_rna))/amp_rna

            diff_rescaled_afd_c = rescaled_afd_rna_c - rescaled_afd_dna_c

            days_range = numpy.linspace(min(days_dna_c), max(days_dna_c), 1000)
            rescaled_days_dna_c = (freq_dna*days_range) + phase_dna
            rescaled_days_rna_c = (freq_rna*days_range) + phase_rna
            sine_diff_prediction = 2*numpy.sin((rescaled_days_rna_c - rescaled_days_dna_c)/2)*numpy.cos((rescaled_days_rna_c + rescaled_days_dna_c)/2)

            ax.plot(days_range, sine_diff_prediction, ls='-', lw=3, c=utils.dna_rna_color_dict['ratio'], zorder=2, label='RNA - DNA sine functions')

            ax.scatter(days_rna_c, diff_rescaled_afd_c, s=8, alpha=1, c=utils.dna_rna_color_dict['ratio'], zorder=1)
            ax.set_xlabel("Time (days)", fontsize=10)
            ax.set_ylabel("CLR-transformed abund., " + utils.rescaled_label_clr_dict['ratio'], fontsize=10)
            ax.set_title(otu_labels_subset[c], fontsize=11)

            #minor_days, major_days, major_labels
            ax.set_xlim([0, max(days_dna_c)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)
            #max_ = numpy.absolute(max(residuals))
            #ax.set_ylim([-1*max_, max_])

            if (chunk_idx == 0) and (c_idx == 0):
                ax.legend(loc='upper right', fontsize=8)


 
    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_clr_ratio.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_time_vs_env():

    param_env_dict = load_param_env_dict()

    env_variable_all_nested = [['water_temp', 'specific_conductivity'], ['dissolved_oxygen', 'salinity'], ['secchi_depth', 'ph']]
    
    
    fig = plt.figure(figsize = (8, 8))
    fig.subplots_adjust(bottom= 0.15)

    for nested_i_idx, nested_i in enumerate(env_variable_all_nested):

        for env_variable_j_idx, env_variable_j in enumerate(nested_i):

            ax = plt.subplot2grid((3, 2), (nested_i_idx, env_variable_j_idx), colspan=1)

            env_variable_array = numpy.asarray([metadata_dict[s][env_variable_j] for s in samples[(sample_type=='RNA')]])
            # remove nans
            env_to_keep_idx = (~numpy.isnan(env_variable_array))
            env_variable_array_clean = env_variable_array[env_to_keep_idx]
            days_clean = days[env_to_keep_idx]

            env_variable_dict_idx = param_env_dict['env_variables_labels'].index(env_variable_j)
            

            ax.scatter(days_clean, env_variable_array_clean, s=5, alpha=1, zorder=2, c='k')
            #ax.set_yscale('log', basey=10)
            #ax.tick_params(axis='both', labelsize=7)

            days_range = numpy.linspace(min(days_clean), max(days_clean), 1000)

            sine_prediction = param_env_dict['amp_leastsq'][env_variable_dict_idx] * numpy.sin(param_env_dict['freq_leastsq'][env_variable_dict_idx] * days_range + param_env_dict['phase_leastsq'][env_variable_dict_idx]) + param_env_dict['param_mean_leastsq'][env_variable_dict_idx]
            ax.plot(days_range, sine_prediction, lw=2, ls='-', alpha=0.5, c='k', zorder=1, label='Sine function')

            ax.set_xlabel('Time (days)', fontsize=9)
            ax.set_ylabel(utils.env_variable_label_dict[env_variable_j], fontsize=9)

            ax.set_xlim([0, max(days_clean)])
            ax.set_xticks(minor_days, minor=True)
            ax.set_xticks(major_days, minor=False)
            ax.set_xticklabels(major_labels, minor=False, fontsize=7)
            ax.yaxis.set_tick_params(labelsize=7)
            ax.xaxis.set_tick_params(labelsize=7)

            if (env_variable_j_idx == 0) and (nested_i_idx==0):
                ax.legend(loc='upper right', fontsize=7)


            print(2*numpy.pi/param_env_dict['freq_leastsq'][env_variable_dict_idx])


    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%stime_vs_env.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")

    # Infer parameters
    #make_param_mle_otu_dict()
    #make_param_env_dict()
    
    plot_time_vs_abundance_clr(data_type='DNA')
    #plot_time_vs_abundance_clr(data_type='RNA')

    # plot includes sine difference
    #plot_time_vs_clr_ratio()

    #plot_time_vs_env()


