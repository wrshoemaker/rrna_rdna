import numpy
from scipy import special, stats
from scipy.special import loggamma, hyperu, digamma, polygamma, gamma

from statsmodels.base.model import GenericLikelihoodModel

import scipy.optimize as so
import mle_utils



def _ll_gamma_param_sampling(n, N, sigma,k):
    # n = exogenous
    # N = endogenous

    x_mean = k*(1-(sigma/2))
    beta = (2-sigma)/sigma

    #ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))
    ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))
    
    return -ll






n = numpy.asarray([3.000e+00, 1.000e+00, 0.000e+00, 9.000e+00, 6.000e+00, 4.000e+00,
                    5.000e+00, 0.000e+00, 1.000e+00, 1.000e+00, 2.000e+00, 1.000e+00,
                    2.000e+00, 3.000e+00, 1.600e+01, 0.000e+00, 0.000e+00, 7.000e+00,
                    0.000e+00, 1.000e+00, 6.000e+00, 1.000e+00, 1.036e+03, 6.400e+01])


N = numpy.asarray([8344, 7107, 8644, 8226, 7104, 7213, 7753, 5525, 8556, 6594, 8805, 8629,
                    8293, 7596, 5507, 3397, 7961, 6312, 7572, 6432, 8435, 7746, 8650, 8557])



#k_start = numpy.mean(n/N)
#sigma_start = 1
#result = so.minimize(_ll_gamma_param_sampling, (1e-6, 1e-6), args=(sigma_start, k_start), bounds=((1e-5,1e5), (k_start,k_start)))

#print(result.x)



#gamma_sampling_model = mle_utils.mle_gamma_sampling_fix_k(N, n)
#gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", start_params=[1], disp = False, bounds= [(0.0000001,1000)])
#gamma_sampling_model_ll = gamma_sampling_model.loglike(gamma_sampling_result.params)

#print(gamma_sampling_result)


