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
import tsdata_to_cpsd


min_coh_xy = 0.3
n_surr=1000


param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
otu_labels = param_dict['otu_labels']
otu_labels.sort()


fig = plt.figure(figsize = (20, 20))
fig.subplots_adjust(bottom= 0.15)

idx_all = list(range(len(otu_labels)))
chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

for chunk_idx, chunk in enumerate(chunk_all):

    for c_idx, c in enumerate(chunk):

        ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

        otu_label_c = otu_labels[c]
        print(otu_label_c)

        otu_c_idx = param_dict['otu_labels'].index(otu_label_c)

        afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_c_idx])
        afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_c_idx])
        rna_dna = numpy.column_stack((afd_rna, afd_dna))

        # channels
        n = rna_dna.shape[1]

        # frequency resolution
        fres = 128
        # MATLAB’s h = fres + 1              
        h = fres + 1 
        fs=1.0
        nfft = 2*(h-1)
        # window length
        window = int(rna_dna.shape[0] / 2)
        noverlap = 30

        S = tsdata_to_cpsd.cpsd_welch_matlab(rna_dna, n=n, h=h, nfft=nfft, window=window, noverlap=noverlap, fs=1.0)
        S_xy = S[0,1,:]

        # Phase spectrum (radians)
        phase_xy = numpy.angle(S_xy)
        # Avoid division by zero at DC
        freqs = numpy.linspace(0, fs/2, h)
        freqs_nonzero = freqs.copy()
        # ignore 0 Hz for lag calculation
        freqs_nonzero[0] = numpy.nan  
        # Time lag at each frequency (same units as 1/fs, e.g., weeks)
        time_lag = phase_xy / (2 * numpy.pi * freqs_nonzero)

        # magnitude-squared coherence
        coh_xy = numpy.abs(S_xy)**2 / (S[0,0,:] * S[1,1,:])
        mask = coh_xy > min_coh_xy
        avg_lag = numpy.nanmean(time_lag[mask])

        print(avg_lag)

        time_lags_null = tsdata_to_cpsd.lag_null_distribution(afd_rna, afd_dna, S.shape, freqs, nfft=nfft, window=window, noverlap=noverlap, fs=fs, n_surr=n_surr, min_coh_xy=min_coh_xy, seed=123456789)

        ax.hist(time_lags_null, bins=20, density=True, histtype='step', alpha=1, lw=3, color='k', zorder=1, label='Null')

        ax.axvline(x=avg_lag, ls=':', c='k', lw=2, zorder=2, label='Observed')
        ax.set_xlabel("Mean lag for CPSD bw RNA and DNA", fontsize=10)
        ax.set_ylabel("Probability density", fontsize=12)
        #ax.set_xlim([-1*max_logfold, max_logfold])
        ax.set_title(otu_label_c, fontsize=12)

        if c == 0:
            ax.legend(loc='upper left', fontsize=10)



fig.subplots_adjust(hspace=0.4, wspace=0.40)
fig_name = "%scpsd_lag_hist.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
