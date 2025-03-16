import config
import sys
import random
import argparse
import copy
import numpy
import utils
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy import stats, signal
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import sine_parameter_utils

# numdifftools also installed
import pickle
from scipy.stats import gamma, loggamma, nbinom


import plot_predict_change_dna


param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
days = numpy.asarray(param_dict['data']['days']['RNA'][0])


numpy.random.seed(123456789)
random.seed(123456789)


focal_otu = 'Otu000001'
#focal_otu_formatted = 'OTU 1'
focal_otu_formatted = 'OTU 1 ('+ r'$\mathit{Anabaena}$' + ' sp.)'
focal_otu_idx = param_dict['otu_labels'].index(focal_otu)

clr_afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][focal_otu_idx])
clr_afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][focal_otu_idx])

diff_clr_afd_dna = clr_afd_dna[1:] - clr_afd_dna[:-1]
diff_clr_afd_rna = clr_afd_rna[1:] - clr_afd_rna[:-1]

rho_0 = numpy.corrcoef(diff_clr_afd_dna, diff_clr_afd_rna)[0,1]

print(0, rho_0)
for n in range(1, 10):

    diff_clr_afd_dna_i = diff_clr_afd_dna[:-n]
    diff_clr_afd_rna_i = diff_clr_afd_dna[n:]

    rho = numpy.corrcoef(diff_clr_afd_dna_i, diff_clr_afd_rna_i)[0,1]

    print(n, rho)




#for 
