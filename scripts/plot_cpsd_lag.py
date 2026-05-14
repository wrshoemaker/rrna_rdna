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


cpsd_dict_path = '%scpsd_dict.pickle' % config.data_directory
param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
otu_labels = param_dict['otu_labels']

# channels
#n = len(param_dict['data']['clr_afd']['DNA'][0])
n = 2
# frequency resolution
fres = 128
# MATLAB’s h = fres + 1              
h = fres + 1 
fs=1.0
nfft = 2*(h-1)
# window length
window = int(len(param_dict['data']['clr_afd']['DNA'][0]) / 2)
noverlap = 30
min_coh_xy = 0.3
n_surr=1000


taxonomy_dict = utils.build_taxonomy_dict()
# target oscillation frequency (approx 0.0195 cycles/week) is at 5th frequency bin.


def make_cpsd_dict(n_null=10000):

    cpsd_dict = {}

    for otu_idx in range(len(param_dict['data']['clr_afd']['DNA'])):

        otu_label_c = otu_labels[otu_idx]

        afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_idx])
        afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_idx])
        rna_dna = numpy.column_stack((afd_rna, afd_dna))

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
        #avg_lag = numpy.nanmean(time_lag[mask])

        mask_scaled = (~numpy.isnan(time_lag))*mask        
        avg_lag_scaled_coh_xy = numpy.real( sum((time_lag[mask_scaled]) * (coh_xy[mask_scaled])) / sum(coh_xy[mask_scaled]))

        time_lags_null, scaled_time_lags_null = tsdata_to_cpsd.lag_null_distribution(afd_rna, afd_dna, S.shape, freqs, nfft=nfft, window=window, noverlap=noverlap, fs=fs, n_surr=n_null, min_coh_xy=min_coh_xy, seed=123456789, n=n)
        p_value_lag = utils.compute_pvalue(avg_lag_scaled_coh_xy, scaled_time_lags_null)

        # amplitude similarity
        # coherence weighted log-ratio of power spectra 
        log_power_ratio = numpy.log(numpy.real(S[0,0,:]) / numpy.real(S[1,1,:]))
        mask_amp = (~numpy.isnan(log_power_ratio)) * mask 
        avg_log_power_ratio = numpy.real(numpy.sum(log_power_ratio[mask_amp] * coh_xy[mask_amp]) / numpy.sum(coh_xy[mask_amp]))

        # timescale oscillation similarity 
        #mask_timescale = ~numpy.isnan(freqs_nonzero)
        #centroid_rna = (numpy.sum(freqs[mask_timescale] * numpy.real(S[0,0,:][mask_timescale])) / numpy.sum(numpy.real(S[0,0,:][mask_timescale])))
        #centroid_dna = (numpy.sum(freqs[mask_timescale] * numpy.real(S[1,1,:][mask_timescale])) / numpy.sum(numpy.real(S[1,1,:][mask_timescale])))
        # ratio of dominant periods
        #timescale_ratio = centroid_dna / centroid_rna
        #log_timescale_ratio = numpy.log(timescale_ratio)
        
        # add coherence mask
        mask_timescale = (~numpy.isnan(freqs_nonzero)) * mask  
        w_rna = numpy.real(S[0,0,:][mask_timescale]) * coh_xy[mask_timescale]
        w_dna = numpy.real(S[1,1,:][mask_timescale]) * coh_xy[mask_timescale]
        centroid_rna = numpy.sum(freqs[mask_timescale] * w_rna) / numpy.sum(w_rna)
        centroid_dna = numpy.sum(freqs[mask_timescale] * w_dna) / numpy.sum(w_dna)
        log_timescale_ratio = numpy.log(centroid_dna / centroid_rna)

        #avg_log_power_ratio_null, timescale_dist_null = tsdata_to_cpsd.amplitude_timescale_null_distribution(afd_rna, afd_dna, S.shape, freqs, nfft, window, noverlap, fs, min_coh_xy=min_coh_xy, n_surr=n_null)
        #avg_log_power_ratio_null, timescale_dist_null = tsdata_to_cpsd.amplitude_timescale_bootstrap(afd_rna, afd_dna, freqs, nfft, window, noverlap, fs, min_coh_xy=min_coh_xy, n_boot=n_null)

        #p_value_power = utils.compute_pvalue(avg_log_power_ratio, avg_log_power_ratio_null)
        #p_value_timescale = utils.compute_pvalue(log_timescale_ratio, timescale_dist_null, side='two')

        #print(p_value_power, p_value_timescale)

        cpsd_dict[otu_label_c] = {}
        cpsd_dict[otu_label_c]['avg_lag_scaled'] = avg_lag_scaled_coh_xy
        cpsd_dict[otu_label_c]['avg_lag_scaled_null'] = scaled_time_lags_null
        cpsd_dict[otu_label_c]['p_value_lag'] = p_value_lag

        cpsd_dict[otu_label_c]['avg_log_power_ratio'] = avg_log_power_ratio
        #cpsd_dict[otu_label_c]['avg_log_power_ratio_null'] = avg_log_power_ratio_null
        #cpsd_dict[otu_label_c]['timescale_ratio_p_value'] = p_value_power

        cpsd_dict[otu_label_c]['log_timescale_ratio'] = log_timescale_ratio
        #cpsd_dict[otu_label_c]['log_timescale_ratio_null'] = timescale_dist_null
        #cpsd_dict[otu_label_c]['p_value_timescale'] = p_value_timescale


    sys.stderr.write("Saving parameter dictionary...\n")
    with open(cpsd_dict_path, 'wb') as outfile:
        pickle.dump(cpsd_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def plot_cpsd_lag():

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]
    cpsd_dict =  pickle.load(open(cpsd_dict_path, 'rb'))

    asv_count = 0

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            otu_label_c = otu_labels[asv_count]
            #otu_c_idx = param_dict['otu_labels'].index(otu_label_c)

            avg_lag_scaled = cpsd_dict[otu_label_c]['avg_lag_scaled']
            avg_lag_scaled_null = cpsd_dict[otu_label_c]['avg_lag_scaled_null']
            p_value_lag = cpsd_dict[otu_label_c]['p_value_lag']

            # convert to days
            mean_delta_days = 7.19672131147541
            avg_lag_scaled_days = avg_lag_scaled*mean_delta_days
            avg_lag_scaled_null_days = avg_lag_scaled_null*mean_delta_days

            counts, bin_edges, patches = ax.hist(avg_lag_scaled_null_days, bins=20, density=True, histtype='step', alpha=1, lw=6, color='k', zorder=2, label='Phase-randomized null')
            max_height = counts.max()*1.1
            max_abs_width = 1.1*numpy.max(numpy.abs(bin_edges))
                
            ax.set_xlim([-1*max_abs_width, max_abs_width])
            ax.set_ylim([0, max_height])

            ax.axvline(x=avg_lag_scaled_days, ls=':', c='k', lw=5, zorder=3, label='Observed')
            ax.set_xlabel("Mean lag time b/w RNA and DNA (d)", fontsize=10)
            ax.set_ylabel("Probability density", fontsize=12)
            #ax.set_xlim([-1*max_logfold, max_logfold])
            ax.set_title('ASV %d (%s)\n' % (asv_count + 1, taxonomy_dict[otu_label_c]['family']) + r'$P=$' + str(p_value_lag), fontsize=12)

            #min_x, max_x = min(avg_lag_scaled_null_days), max(avg_lag_scaled_null_days)
            lag_range = numpy.linspace(-1*max_abs_width, max_abs_width, num=10000)
            ax.fill_between(lag_range, max_height, where=lag_range > 0, facecolor= utils.dna_rna_color_dict['RNA'], alpha=0.5, zorder=1)
            ax.fill_between(lag_range,max_height, where=lag_range < 0, facecolor= utils.dna_rna_color_dict['DNA'], alpha=0.5, zorder=1)

            if c == 0:
                ax.legend(loc='upper left', fontsize=10)

            asv_count += 1


    fig.subplots_adjust(hspace=0.4, wspace=0.40)
    fig_name = "%scpsd_lag_hist.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def test_significance_cpsd_all_otus():

    cpsd_dict =  pickle.load(open(cpsd_dict_path, 'rb'))
    # one-sample wilcoxon signed-rank test 
    all_stats = ['avg_lag_scaled', 'avg_log_power_ratio', 'log_timescale_ratio']
    for s in all_stats:

        stat = numpy.asarray([cpsd_dict[o][s] for o in cpsd_dict.keys()] )

        stat = numpy.real(stat)
        print(stat)

        w, p = stats.wilcoxon(stat)

        print(s, w, p)




if __name__ == "__main__":

    print("Running...")

    #make_cpsd_dict()
    #plot_cpsd_lag()

    test_significance_cpsd_all_otus()