import copy
import numpy
from scipy.special import loggamma
from lmfit import Minimizer, create_params


afd_clean = numpy.asarray([0.1, 0.05, 0.04, 0.01212121])
days_afd_clean = numpy.asarray([10, 30, 50, 70])



freq_value = 2*numpy.pi/365 # 0.01721420632
freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
freq_max = 2*numpy.pi/70 # 0.034906585 (365-185)


# this is an extra parameter I inferred seperately. 
beta_estimate = 1.4

phase_value = numpy.pi
phase_min = 0
phase_max = 2*numpy.pi

param_mean_value = numpy.mean(afd_clean)
param_min_value = min(afd_clean)
param_max_value = max(afd_clean)

amp_value = 1
amp_min = 0.001
amp_max = 10


# initialize the parameter initial values as well as their bounds (max and min)
# amp, freq, phase, and param_mean are the parameters for my particular model

params = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                            freq=dict(value=freq_value, min=freq_min, max=freq_max),
                            phase=dict(value=phase_value, min=phase_min, max=phase_max),
                            param_mean=dict(value=param_mean_value, min=param_min_value, max=param_max_value))


# log likelihood function for my gamma distribution with oscillating carrying capacity
# We are using a *minimize* function so in order to maximize the LL we have to minimize the negative LL
def ll_sine_gamma(params, days_afd, afd, beta):

    
    amp = params['amp']
    freq = params['freq']
    phase = params['phase']
    param_mean = params['param_mean']

    x_bar_pred = numpy.exp(amp*(numpy.cos(phase)*numpy.sin(freq*days_afd) + numpy.sin(phase)*numpy.cos(freq*days_afd))) * param_mean
    
    ll = (beta-1)*sum(numpy.log(afd)) - beta*sum(afd/x_bar_pred) - beta*sum(numpy.log(x_bar_pred)) + len(afd)*beta*numpy.log(beta) - len(afd)*loggamma(beta)

    return -1*ll


def grid_search_mle_sine_wave(days_afd_, afd_, params_, beta_estimate):

    # minimize the negative log-likelihood 
    fitter = Minimizer(ll_sine_gamma, params_, fcn_args=(days_afd_, afd_, beta_estimate))
    
    # brute force results
    result_brute = fitter.minimize(method='brute', Ns=30, keep=25)

    return result_brute, fitter


def second_round_optimization_mle(result_brute, fitter, beta):

    # second round of optimization using least-squares with brute force as a starting point
    best_result_leastsq = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:
        trial = fitter.minimize(method='lbfgsb', params=candidate.params)
        if trial.chisqr < best_result_leastsq.chisqr:
            best_result_leastsq = trial

    return best_result_leastsq


# Step 1: perform a grid search over parameter combinations
result_brute, fitter = grid_search_mle_sine_wave(days_afd_clean, afd_clean, params, beta_estimate)

# Step 2: optimize using the grid value with the smallest negative LL 
best_result_mle = second_round_optimization_mle(result_brute, fitter, beta_estimate)
# get the parameters
best_params_mle = best_result_mle.params

