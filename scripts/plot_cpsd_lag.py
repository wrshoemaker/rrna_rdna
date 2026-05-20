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
mean_delta_days = 7.19672131147541

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


        #frequency-specific threshold from null
        # C_thresh is shape (h,) => one threshold per frequency bin
        _, C_null, C_thresh = tsdata_to_cpsd.phase_randomized_coherence_null(afd_rna, afd_dna, fs=fs, window=window, noverlap=noverlap, nfft=nfft, n_surr=1000)
        
        freqs_nonzero = freqs.copy()
        freqs_nonzero[0] = numpy.nan
        
         # radians, shape (h,)
        phase_xy  = numpy.angle(S[0, 1, :])
        phase_deg = numpy.degrees(phase_xy)

        # mask to significant frequencies only, same mask as lag calculation
        #sig_mask = (coh_xy > C_thresh) & (~numpy.isnan(freqs_nonzero))

        cpsd_dict[otu_label_c]['C_thresh'] = C_thresh
        cpsd_dict[otu_label_c]['coh_xy'] = coh_xy
        cpsd_dict[otu_label_c]['freqs'] = freqs

        # phase plot
        phase_xy = numpy.degrees(numpy.angle(S[0, 1, :]))
        cpsd_dict[otu_label_c]['phase_xy'] = phase_xy


        # dominant frequencies from parametric fit
        #afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_idx])
        #afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_idx])

        #tau_rna = 2*numpy.pi/param_dict['freq_mle']['RNA'][otu_idx]
        #tau_dna = 2*numpy.pi/param_dict['freq_mle']['DNA'][otu_idx]
        
        #f_rna = 1.0 / tau_rna
        #f_dna = 1.0 / tau_dna
        #idx_rna = numpy.argmin(numpy.abs(freqs - f_rna))
        #idx_dna = numpy.argmin(numpy.abs(freqs - f_dna))

        #phase_at_rna = phase_deg[idx_rna]
        #phase_at_dna = phase_deg[idx_dna]
        ##coh_at_rna   = coh_xy[idx_rna]
        #coh_at_dna   = coh_xy[idx_dna]



    sys.stderr.write("Saving parameter dictionary...\n")
    with open(cpsd_dict_path, 'wb') as outfile:
        pickle.dump(cpsd_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def test_cpsd_lag():

    days = numpy.asarray(param_dict['data']['days']['DNA'][0])
    afd_dna = numpy.sin(days)
    # RNA LAGS DNA
    # DNA -> RNA
    afd_rna = numpy.sin(days - 5) 

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


    print(avg_lag_scaled_coh_xy)




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
            avg_lag_scaled_days = avg_lag_scaled*mean_delta_days
            avg_lag_scaled_null_days = avg_lag_scaled_null*mean_delta_days

            counts, bin_edges, patches = ax.hist(avg_lag_scaled_null_days, bins=20, density=True, histtype='step', alpha=1, lw=6, color='k', zorder=2, label='Phase-randomized null')
            max_height = counts.max()*1.1
            max_abs_width = 1.1*numpy.max(numpy.abs(bin_edges))
                
            ax.set_xlim([-1*max_abs_width, max_abs_width])
            ax.set_ylim([0, max_height])

            ax.axvline(x=avg_lag_scaled_days, ls=':', c='k', lw=5, zorder=3, label='Observed')
            #ax.set_xlabel("Mean lag time b/w RNA and DNA (d)", fontsize=10)
            ax.set_xlabel("Mean cross-spectral phase delay (days), " + r'$\overline{\mathcal{T}} $', fontsize=11)

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



def plot_cpsd_lag_global():

    cpsd_dict =  pickle.load(open(cpsd_dict_path, 'rb'))
    otu_label_all = list(cpsd_dict.keys())

    n_iter = len(cpsd_dict[otu_label_all[0]]['avg_lag_scaled_null'])
    avg_lag_scaled_avg = numpy.mean([ cpsd_dict[o]['avg_lag_scaled'] for o in otu_label_all])
    avg_lag_scaled_null_avg = numpy.asarray([numpy.mean([cpsd_dict[o]['avg_lag_scaled_null'][i] for o in otu_label_all]) for i in range(n_iter)])

    avg_lag_scaled_avg_days = avg_lag_scaled_avg*mean_delta_days
    avg_lag_scaled_null_avg_days = avg_lag_scaled_null_avg*mean_delta_days

    p_value = utils.compute_pvalue(avg_lag_scaled_avg, avg_lag_scaled_null_avg)

    print(p_value)
    
    fig, ax = plt.subplots(figsize=(4.5,4))
    counts, bin_edges, patches = ax.hist(avg_lag_scaled_null_avg_days, bins=20, density=True, histtype='step', alpha=1, lw=6, color='k', zorder=2, label='Phase-randomized null')

    max_height = counts.max()*1.1
    max_abs_width = 1.1*numpy.abs(avg_lag_scaled_avg_days)

    ax.axvline(x=avg_lag_scaled_avg_days, ls=':', c='k', lw=5, zorder=3, label='Observed')
    ax.set_xlabel("Mean cross-spectral phase delay (days), " + r'$\overline{\mathcal{T}_{\mathrm{global}}}$', fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)

    ax.set_xlim([-1*max_abs_width, max_abs_width])
    ax.set_ylim([0, max_height])

    ax.text(0.76, 0.85, 'rRNA ' + r'$\rightarrow$' + ' rDNA', fontsize=13, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.25, 0.85, 'rDNA ' + r'$\rightarrow$' + ' rRNA', fontsize=13, ha='center', va='center', transform=ax.transAxes)

    ax.text(0.76, 0.75, r'$P_{\text{global}} < 10^{-4} $', fontsize=12, ha='center', va='center', transform=ax.transAxes)


    lag_range = numpy.linspace(-1*max_abs_width, max_abs_width, num=10000)
    ax.fill_between(lag_range, max_height, where=lag_range > 0, facecolor= utils.dna_rna_color_dict['RNA'], alpha=0.5, zorder=1)
    ax.fill_between(lag_range,max_height, where=lag_range < 0, facecolor= utils.dna_rna_color_dict['DNA'], alpha=0.5, zorder=1)
    
    #ax.legend(loc='upper left', fontsize=10)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncols=3, borderaxespad=0)

    fig.subplots_adjust(hspace=0.4, wspace=0.40)
    fig_name = "%scpsd_lag_hist_global.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_coherence_spectrum(alpha=0.05):

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

            C_thresh = cpsd_dict[otu_label_c]['C_thresh']
            coh_xy = cpsd_dict[otu_label_c]['coh_xy']
            freqs = cpsd_dict[otu_label_c]['freqs']
            tau_rna = 2*numpy.pi/param_dict['freq_mle']['RNA'][asv_count]
            tau_dna = 2*numpy.pi/param_dict['freq_mle']['DNA'][asv_count]

            ax.plot(freqs, coh_xy, label='coherence')
            ax.plot(freqs, C_thresh, 'k--', label=f'null (p={alpha})')
            ax.fill_between(freqs, 0, coh_xy, where=coh_xy > C_thresh, alpha=0.3, label='significant')

            if tau_rna is not None:
                ax.axvline(1/tau_rna, color='r', lw=0.8, label=f'1/τ_rna')
            if tau_dna is not None:
                ax.axvline(1/tau_dna, color='b', lw=0.8, label=f'1/τ_dna')

            ax.set_xlabel('frequency')
            ax.set_ylabel('coherence')


            asv_count += 1


    fig.subplots_adjust(hspace=0.4, wspace=0.40)
    fig_name = "%scpsd_coherence_spectrum.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_phase_spectrum():

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    idx_all = list(range(len(otu_labels)))
    chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]
    cpsd_dict =  pickle.load(open(cpsd_dict_path, 'rb'))

    asv_count = 0

    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):


            otu_label_c = otu_labels[asv_count]

            C_thresh = cpsd_dict[otu_label_c]['C_thresh']
            coh_xy = cpsd_dict[otu_label_c]['coh_xy']
            phase_xy = cpsd_dict[otu_label_c]['phase_xy']
            freqs = cpsd_dict[otu_label_c]['freqs']
            
            sig_mask = coh_xy > C_thresh
            phase_masked = numpy.where(sig_mask, phase_xy, numpy.nan)

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            ax.plot(freqs, phase_masked)
            ax.axhline(0,    color='k', lw=0.8, ls='--')
            ax.axhline(180,  color='k', lw=0.5, ls=':')
            ax.axhline(-180, color='k', lw=0.5, ls=':')
            
            tau_rna = 2*numpy.pi/param_dict['freq_mle']['RNA'][asv_count]
            tau_dna = 2*numpy.pi/param_dict['freq_mle']['DNA'][asv_count]


            if tau_rna is not None:
                idx = numpy.argmin(numpy.abs(freqs - 1/tau_rna))
                ax.axvline(1/tau_rna, color='r', lw=0.8, label=f'1/τ_rna  φ={phase_masked[idx]:.1f}°')
            if tau_dna is not None:
                idx = numpy.argmin(numpy.abs(freqs - 1/tau_dna))
                ax.axvline(1/tau_dna, color='b', lw=0.8, label=f'1/τ_dna  φ={phase_masked[idx]:.1f}°')

            ax.set_xlabel('frequency')
            ax.set_ylabel('phase (degrees)')
            ax.set_ylim(-180, 180)


    fig.subplots_adjust(hspace=0.4, wspace=0.40)
    fig_name = "%scpsd_coherence_spectrum.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")

    #make_cpsd_dict()
    #plot_cpsd_lag()

    #test_significance_cpsd_all_otus()
    plot_cpsd_lag_global()

    #test_cpsd_lag()

    #plot_phase_spectrum()