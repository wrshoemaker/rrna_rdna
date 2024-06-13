import random
import copy
import sys
import numpy
import random
import pickle
import scipy.stats as stats

from scipy.special import loggamma, hyperu, digamma, polygamma, gamma

import matplotlib.pyplot as plt
import functools
import operator

import config
import mpmath

from statsmodels.base.model import GenericLikelihoodModel


numpy.random.seed(123456789)


def _ll_gamma_sampling(n, N, x_mean, x_var):
    # n = exogenous
    # N = endogenous

    beta = (x_mean**2)/x_var
    #ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))
    ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))

    return ll


def _ll_gamma_param_sampling(n, N, K, sigma):
    # n = exogenous
    # N = endogenous

    x_mean = K*(1-(sigma/2))
    beta = (2-sigma)/sigma

    #ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))
    ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))

    return ll



def _ll_gamma_param_sine_sampling(n_and_t, N, K, sigma, amp, freq, phase):
    # n = exogenous
    # N = endogenous

    n, t = n_and_t[:,0], n_and_t[:,1]

    x_mean = K*numpy.exp(amp*numpy.sin(t*freq + phase))
    beta = (2-sigma)/sigma

    #ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))
    ll = loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))

    return ll



class mle_gamma_sampling(GenericLikelihoodModel):
    
    def __init__(self, endog, exog, **kwds):
        super(mle_gamma_sampling, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        x_mean = params[0]
        x_var = params[1]
        ll = -1*_ll_gamma_sampling(self.exog.flatten(), self.endog, x_mean, x_var)
        return ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        #print(type(start_params).__module__, numpy.__name__ )
        #if (type(start_params).__module__ == numpy.__name__ ) == False:

        if type(start_params) == type(None):
            x_mean_start = 0.001
            x_var_start = 0.0001
            start_params = numpy.array([x_mean_start, x_var_start])


        return super(mle_gamma_sampling, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)



class mle_gamma_param_sampling(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(mle_gamma_param_sampling, self).__init__(endog, exog, **kwds)


    def nloglikeobs(self, params):
        k = params[0]
        sigma = params[1]
        ll = -1*_ll_gamma_param_sampling(self.exog.flatten(), self.endog, k, sigma)
        return ll


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if type(start_params) == type(None):
            k_0 = 0.001
            sigma_0 = 0.0001
            start_params = numpy.array([k_0, sigma_0])


        return super(mle_gamma_param_sampling, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)




class mle_gamma_param_sine_sampling(GenericLikelihoodModel):
    
    def __init__(self, endog, exog, **kwds):
        super(mle_gamma_param_sine_sampling, self).__init__(endog, exog, **kwds)


    def nloglikeobs(self, params):
        k = params[0]
        sigma = params[1]
        amp = params[2]
        freq = params[3]
        phase = params[4]
        ll = -1*_ll_gamma_param_sine_sampling(self.exog, self.endog, k, sigma, amp, freq, phase)
        return ll


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        if type(start_params) == type(None):
            k_0 = 0.001
            sigma_0 = 0.0001
            amp_0 = 0.5
            freq_0 = 0.4
            phase_0 = numpy.pi
            start_params = numpy.array([k_0, sigma_0, amp_0, freq_0, phase_0])


        return super(mle_gamma_param_sine_sampling, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)






def test_mle():

    #n = numpy.asarray([24,30,30,28,4,0,24,21,20,19,0,22,24,22,23,25,26,19,20,19,18,16])
    #N = numpy.asarray([1000]* len(n))

    n = numpy.asarray([3.000e+00, 1.000e+00, 0.000e+00, 9.000e+00, 6.000e+00, 4.000e+00,
                    5.000e+00, 0.000e+00, 1.000e+00, 1.000e+00, 2.000e+00, 1.000e+00,
                    2.000e+00, 3.000e+00, 1.600e+01, 0.000e+00, 0.000e+00, 7.000e+00,
                    0.000e+00, 1.000e+00, 6.000e+00, 1.000e+00, 1.036e+03, 6.400e+01])


    N = numpy.asarray([8344, 7107, 8644, 8226, 7104, 7213, 7753, 5525, 8556, 6594, 8805, 8629,
                        8293, 7596, 5507, 3397, 7961, 6312, 7572, 6432, 8435, 7746, 8650, 8557])
    
    t = numpy.asarray(range(len(n)))

    n_and_t = numpy.transpose((n, t))
    


    mu_start = numpy.mean(n/N)
    sigma_start = numpy.std(n/N)
    start_params = numpy.asarray([mu_start, sigma_start])

    #gamma_sampling_model = mle_gamma_sampling(N, n)
    #gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", start_params=start_params, disp=False, bounds= [(mu_start,mu_start), (0.000001,1000)])
    #gamma_sampling_model_ll = gamma_sampling_model.loglike(gamma_sampling_result.params)

    #print(gamma_sampling_result.params)

    gamma_param_sampling_model = mle_gamma_param_sampling(N, n)
    gamma_param_sampling_result = gamma_param_sampling_model.fit(method="lbfgs", start_params=start_params, disp = True, bounds= [(0.0001,1), (0.001,1.9999)])
    gamma_param_sampling_model_ll = gamma_param_sampling_model.loglike(gamma_param_sampling_result.params)

    #print(gamma_param_sampling_result.mle_retvals)
    #print(start_params)
    #print(gamma_param_sampling_result.params)
    
    #start_params = numpy.asarray([mu_start, sigma_start, 1.8, numpy.pi, numpy.pi])

    #gamma_param_sine_sampling_model = mle_gamma_param_sine_sampling(N, n_and_t)
    #gamma_param_sine_sampling_result = gamma_param_sine_sampling_model.fit(method="lbfgs", start_params=start_params, disp = False, bounds= [(mu_start,mu_start), (0.000001,1000), (0.0001, 10), (0,2*numpy.pi), (0,2*numpy.pi)])
    #gamma_param_sine_sampling_ll = gamma_param_sine_sampling_model.loglike(gamma_param_sine_sampling_result.params)
    





if __name__ == "__main__":

    test_mle()