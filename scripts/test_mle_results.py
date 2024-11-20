import config
import numpy
import utils
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, colors

from scipy import stats

import sine_parameter_utils

import pickle

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))


data_type = 'DNA'


range_ = range(len(param_dict['data']['days'][data_type]))

for i in range_:

    clr_afd = numpy.asarray(param_dict['data']['clr_afd'][data_type][i])
    days_afd = numpy.asarray(param_dict['data']['days'][data_type][i])
    afd = numpy.exp(clr_afd)

    amp = param_dict['amp_mle'][data_type][i]
    freq = param_dict['freq_mle'][data_type][i]
    phase = param_dict['phase_mle'][data_type][i]
    param_mean = param_dict['param_mean_mle'][data_type][i]
    beta = param_dict['beta'][data_type][i]


    x_bar_pred = numpy.exp(amp*(numpy.cos(phase)*numpy.sin(freq*days_afd) + numpy.sin(phase)*numpy.cos(freq*days_afd))) * param_mean

    c = sum(afd/x_bar_pred) + sum(numpy.log(x_bar_pred/beta))
    print(amp, beta, c)