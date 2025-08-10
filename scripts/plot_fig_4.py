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
from scipy.stats import gamma, loggamma, nbinom, norm


import plot_predict_change_dna


slope_delta_null_dict_path = config.data_directory + 'slope_delta_null_dict.pickle'



numpy.random.seed(123456789)
random.seed(123456789)


method = 'mle'

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
null_predict_change_dict = plot_predict_change_dna.load_null_predict_change_dict_path()
days = numpy.asarray(param_dict['data']['days']['RNA'][0])


focal_otu = 'Otu000001'
#focal_otu_formatted = 'OTU 1'
focal_otu_formatted = 'OTU 1 ('+ r'$\mathit{Anabaena}$' + ' sp.)'
focal_otu_idx = param_dict['otu_labels'].index(focal_otu)

clr_afd_dna = numpy.asarray(param_dict['data']['clr_afd']['DNA'][focal_otu_idx])
clr_afd_rna = numpy.asarray(param_dict['data']['clr_afd']['RNA'][focal_otu_idx])

amp_dna = param_dict['amp_%s' % method]['DNA'][focal_otu_idx]
amp_rna = param_dict['amp_%s' % method]['RNA'][focal_otu_idx]

freq_dna = param_dict['freq_%s' % method]['DNA'][focal_otu_idx]
freq_rna = param_dict['freq_%s' % method]['RNA'][focal_otu_idx]

phase_dna = param_dict['phase_%s' % method]['DNA'][focal_otu_idx]
phase_rna = param_dict['phase_%s' % method]['RNA'][focal_otu_idx]

param_mean_dna = param_dict['param_mean_%s' % method]['DNA'][focal_otu_idx]
param_mean_rna = param_dict['param_mean_%s' % method]['RNA'][focal_otu_idx]

sigma_dna =  param_dict['sigma']['DNA'][focal_otu_idx]
sigma_rna =  param_dict['sigma']['RNA'][focal_otu_idx]

mean_dna = param_mean_dna * numpy.exp(amp_dna * numpy.sin(freq_dna*days + phase_dna))
mean_rna = param_mean_rna * numpy.exp(amp_rna * numpy.sin(freq_rna*days + phase_rna))

k_dna = mean_dna / (1 - (sigma_dna/2))
k_rna = mean_rna / (1 - (sigma_rna/2))



def sine_slope_delta_null_one_iter(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=0):

    Z = numpy.random.multivariate_normal([0, 0], [[1, rhogamma], [rhogamma, 1]], len(days))
    U = norm.cdf(Z)

    exp_clr_sim_dna = gamma.ppf(U[:,0], numpy.divide(2,sigma_dna)-1, scale=sigma_dna*k_dna/2)
    exp_clr_sim_rna = gamma.ppf(U[:,1], numpy.divide(2,sigma_rna)-1, scale=sigma_rna*k_rna/2)

    clr_sim_dna = numpy.log(exp_clr_sim_dna)
    clr_sim_rna = numpy.log(exp_clr_sim_rna)

    diff_rna = clr_sim_rna[1:] - clr_sim_rna[:-1]
    diff_dna = clr_sim_dna[1:] - clr_sim_dna[:-1]

    diff_days = days[1:] - days[:-1]

    diff_rna = diff_rna/diff_days
    diff_dna = diff_dna/diff_days

    slope, intercept, r_value, p_value, std_err = stats.linregress(diff_dna, diff_rna)

    return slope, r_value



def sine_slope_delta_null(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=0, n_iter=int(1e4)):

    slope_all = []
    r_value_all = []
    for n in range(n_iter):

        slope_n, r_value_n = sine_slope_delta_null_one_iter(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=rhogamma)

        slope_all.append(slope_n)
        r_value_all.append(r_value_n)

    slope_all = numpy.asarray(slope_all)
    r_value_all = numpy.asarray(r_value_all)

    return slope_all, r_value_all




def sine_slope_ratio_vs_dna_null(k_dna, k_rna, sigma_dna, sigma_rna, n_iter = int(1e4)):

    slope_all = []
    for n in range(n_iter):

        exp_clr_sim_dna = gamma.rvs(numpy.divide(2,sigma_dna)-1, scale=sigma_dna*k_dna/2, size=len(days))
        exp_clr_sim_rna = gamma.rvs(numpy.divide(2,sigma_rna)-1, scale=sigma_rna*k_rna/2, size=len(days))

        clr_sim_dna = numpy.log(exp_clr_sim_dna)
        clr_sim_rna = numpy.log(exp_clr_sim_rna)

        diff_rna_dna = (clr_sim_rna - clr_sim_dna)[:-1]

        #diff_rna = clr_sim_rna[1:] - clr_sim_rna[:-1]
        diff_dna = clr_sim_dna[1:] - clr_sim_dna[:-1]

        slope, intercept, r_value, p_value, std_err = stats.linregress(diff_rna_dna, diff_dna)

        slope_all.append(slope)


    slope_all = numpy.asarray(slope_all)

    return slope_all



def sine_slope_ratio_vs_dna_w_corr_null(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=0, n_iter=int(1e4)):

    slope_all = []
    r_value_all = []
    for n in range(n_iter):

        Z = numpy.random.multivariate_normal([0, 0], [[1, rhogamma], [rhogamma, 1]], len(days))
        U = norm.cdf(Z)

        exp_clr_sim_dna = gamma.ppf(U[:,0], numpy.divide(2,sigma_dna)-1, scale=sigma_dna*k_dna/2)
        exp_clr_sim_rna = gamma.ppf(U[:,1], numpy.divide(2,sigma_rna)-1, scale=sigma_rna*k_rna/2)

        clr_sim_dna = numpy.log(exp_clr_sim_dna)
        clr_sim_rna = numpy.log(exp_clr_sim_rna)

        diff_rna_dna = (clr_sim_rna - clr_sim_dna)[:-1]

        diff_dna = (clr_sim_dna[1:] - clr_sim_dna[:-1])/(days[1:] - days[:-1])

        slope, intercept, r_value, p_value, std_err = stats.linregress(diff_rna_dna, diff_dna)

        slope_all.append(slope)
        r_value_all.append(r_value)


    slope_all = numpy.asarray(slope_all)
    r_value_all = numpy.asarray(r_value_all)

    return slope_all, r_value_all



def misc_():

    for focal_otu in param_dict['otu_labels']:

        continue

        focal_otu_idx = param_dict['otu_labels'].index(focal_otu)

        # parameters for OTU1
        amp_dna = param_dict['amp_%s' % method]['DNA'][focal_otu_idx]
        amp_rna = param_dict['amp_%s' % method]['RNA'][focal_otu_idx]

        freq_dna = param_dict['freq_%s' % method]['DNA'][focal_otu_idx]
        freq_rna = param_dict['freq_%s' % method]['RNA'][focal_otu_idx]

        phase_dna = param_dict['phase_%s' % method]['DNA'][focal_otu_idx]
        phase_rna = param_dict['phase_%s' % method]['RNA'][focal_otu_idx]

        param_mean_dna = param_dict['param_mean_%s' % method]['DNA'][focal_otu_idx]
        param_mean_rna = param_dict['param_mean_%s' % method]['RNA'][focal_otu_idx]

        sigma_dna =  param_dict['sigma']['DNA'][focal_otu_idx]
        sigma_rna =  param_dict['sigma']['RNA'][focal_otu_idx]

        mean_dna = param_mean_dna * numpy.exp(amp_dna * numpy.sin(freq_dna*days + phase_dna))
        mean_rna = param_mean_rna * numpy.exp(amp_rna * numpy.sin(freq_rna*days + phase_rna))

        k_dna = mean_dna / (1 - (sigma_dna/2))
        k_rna = mean_rna / (1 - (sigma_rna/2))

        sine_slope_null_i = sine_slope_null(k_dna, k_rna, sigma_dna, sigma_rna)

        clr_s_by_s_rescaled_ratio = numpy.asarray(null_predict_change_dict[focal_otu]['clr_s_by_s_rescaled_ratio'])
        diff_clr_s_by_s_rescaled_dna = numpy.asarray(null_predict_change_dict[focal_otu]['diff_clr_s_by_s_rescaled_dna'])
        diff_clr_s_by_s_rescaled_rna = numpy.asarray(null_predict_change_dict[focal_otu]['diff_clr_s_by_s_rescaled_rna'])

        slope, intercept, r_value, p_value, std_err = stats.linregress(diff_clr_s_by_s_rescaled_dna, diff_clr_s_by_s_rescaled_rna)
        #print(slope, stats.linregress(diff_clr_s_by_s_rescaled_rna, diff_clr_s_by_s_rescaled_dna)[0])

        p_value = sum(sine_slope_null_i < slope)/len(sine_slope_null_i)

        print(focal_otu, intercept, slope, p_value)




def plot_ratio_vs_dna_():

    amp_dna = param_dict['amp_%s' % method]['DNA'][focal_otu_idx]
    amp_rna = param_dict['amp_%s' % method]['RNA'][focal_otu_idx]

    freq_dna = param_dict['freq_%s' % method]['DNA'][focal_otu_idx]
    freq_rna = param_dict['freq_%s' % method]['RNA'][focal_otu_idx]

    phase_dna = param_dict['phase_%s' % method]['DNA'][focal_otu_idx]
    phase_rna = param_dict['phase_%s' % method]['RNA'][focal_otu_idx]

    param_mean_dna = param_dict['param_mean_%s' % method]['DNA'][focal_otu_idx]
    param_mean_rna = param_dict['param_mean_%s' % method]['RNA'][focal_otu_idx]

    sigma_dna =  param_dict['sigma']['DNA'][focal_otu_idx]
    sigma_rna =  param_dict['sigma']['RNA'][focal_otu_idx]

    mean_dna = param_mean_dna * numpy.exp(amp_dna * numpy.sin(freq_dna*days + phase_dna))
    mean_rna = param_mean_rna * numpy.exp(amp_rna * numpy.sin(freq_rna*days + phase_rna))

    k_dna = mean_dna / (1 - (sigma_dna/2))
    k_rna = mean_rna / (1 - (sigma_rna/2))

    rhogamma_all = [-0.9, -0.7, 0.7, 0.9, 0.95, 0.97, 0.99]
    for rhogamma in rhogamma_all:

        sine_slope_null_i = sine_slope_ratio_vs_dna_w_corr_null(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=rhogamma)

        print(rhogamma, numpy.mean(sine_slope_null_i))
    

    clr_s_by_s_rescaled_ratio = numpy.asarray(null_predict_change_dict[focal_otu]['clr_s_by_s_rescaled_ratio'])
    diff_clr_s_by_s_rescaled_dna = numpy.asarray(null_predict_change_dict[focal_otu]['diff_clr_s_by_s_rescaled_dna'])
    #diff_clr_s_by_s_rescaled_rna = numpy.asarray(null_predict_change_dict[focal_otu]['diff_clr_s_by_s_rescaled_rna'])

    slope, intercept, r_value, p_value, std_err = stats.linregress(clr_s_by_s_rescaled_ratio, diff_clr_s_by_s_rescaled_dna)

    p_value = sum(sine_slope_null_i > slope)/len(sine_slope_null_i)

    print(slope, sine_slope_null_i[:10])

    #print(1-p_value)

    slope_null_time_all = []
    idx_ = numpy.arange(len(clr_afd_dna))

    for n in range(1000):

        numpy.random.shuffle(idx_)
        #numpy.random.shuffle(clr_afd_rna)

    #    clr_afd_dna_n = clr_afd_dna[idx_]
    #    clr_afd_rna_n = clr_afd_rna[idx_]

    #    diff_clr_afd_dna_null = clr_afd_dna_n[1:] - clr_afd_dna_n[:-1]
    #    diff_clr_afd_rna_null = clr_afd_rna_n[1:] - clr_afd_rna_n[:-1]

    #    slope_null_time_all.append(stats.linregress(diff_clr_afd_dna_null, diff_clr_afd_rna_null)[0])




def old_plot():


    fig = plt.figure(figsize = (8.5, 4)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=1, ncols=2)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_sine_null = fig.add_subplot(gs[0, 1])


    #n_reads_dna_total = numpy.asarray(null_predict_change_dict[focal_otu]['n_reads_dna_occupancy_total'])
    #n_reads_rna_total = numpy.asarray(null_predict_change_dict[focal_otu]['n_reads_rna_occupancy_total'])

    #print(n_reads_rna_total)

    #ax_scatter.scatter(clr_s_by_s_rescaled_ratio, diff_clr_s_by_s_rescaled_dna, s=8, alpha=1, c='k', zorder=2)
    ax_scatter.scatter(diff_clr_s_by_s_rescaled_dna, diff_clr_s_by_s_rescaled_rna, s=8, alpha=1, c='k', zorder=2)
    #ax_scatter.set_xlabel("Difference between RNA and DNA at time " + r'$t$', fontsize=10)
    ax_scatter.set_xlabel("Change in DNA between " + r'$\delta t$' + ' and ' + r'$t+\delta t$', fontsize=10)
    ax_scatter.set_ylabel("Change in RNA between " + r'$\delta t$' + ' and ' + r'$t+\delta t$', fontsize=10)
    ax_scatter.set_title( focal_otu_formatted + '\nNull: Time label permutation', fontsize=11)

    #slope, intercept, r_value, p_value, std_err = stats.linregress(clr_s_by_s_rescaled_ratio, diff_clr_s_by_s_rescaled_dna)
    slope, intercept, r_value, p_value, std_err = stats.linregress(diff_clr_s_by_s_rescaled_dna, diff_clr_s_by_s_rescaled_rna)

    x_range_ =  numpy.linspace(min(diff_clr_s_by_s_rescaled_dna), max(diff_clr_s_by_s_rescaled_dna), 10000)
    y_fit_range = slope*x_range_ + intercept
    ax_scatter.plot(x_range_, y_fit_range, ls='--', lw=2.5, c='k')

    ax_scatter.text(0.26, 0.78, r'$P = $' + str(round(p_value, 5)), fontsize=12, ha='center', va='center', transform=ax_scatter.transAxes)
    #ax_scatter.text(0.26, 0.78, utils.get_p_value_latex_label_dict(p_value), fontsize=12, ha='center', va='center', transform=ax_scatter.transAxes)
    ax_scatter.text(0.26, 0.87, 'Slope = ' + str(round(slope, 3)), fontsize=12, ha='center', va='center', transform=ax_scatter.transAxes)

    amp_dna = param_dict['amp_%s' % method]['DNA'][focal_otu_idx]
    amp_rna = param_dict['amp_%s' % method]['RNA'][focal_otu_idx]

    freq_dna = param_dict['freq_%s' % method]['DNA'][focal_otu_idx]
    freq_rna = param_dict['freq_%s' % method]['RNA'][focal_otu_idx]

    phase_dna = param_dict['phase_%s' % method]['DNA'][focal_otu_idx]
    phase_rna = param_dict['phase_%s' % method]['RNA'][focal_otu_idx]

    param_mean_dna = param_dict['param_mean_%s' % method]['DNA'][focal_otu_idx]
    param_mean_rna = param_dict['param_mean_%s' % method]['RNA'][focal_otu_idx]

    sigma_dna =  param_dict['sigma']['DNA'][focal_otu_idx]
    sigma_rna =  param_dict['sigma']['RNA'][focal_otu_idx]

    #sigma_dna = 1.8
    #sigma_rna = 1.8

    mean_dna = param_mean_dna * numpy.exp(amp_dna * numpy.sin(freq_dna*days + phase_dna))
    mean_rna = param_mean_rna * numpy.exp(amp_rna * numpy.sin(freq_rna*days + phase_rna))

    k_dna = mean_dna / (1 - (sigma_dna/2))
    k_rna = mean_rna / (1 - (sigma_rna/2))

    sine_slope_null_i = sine_slope_null(k_dna, k_rna, sigma_dna, sigma_rna)

    p_value_null_model = sum(sine_slope_null_i < slope)/len(sine_slope_null_i)

    print(p_value_null_model)

    ax_sine_null.text(0.76, 0.72, r'$P = $' + str(round(p_value_null_model, 3)), fontsize=12, ha='center', va='center', transform=ax_sine_null.transAxes)


    ax_sine_null.hist(sine_slope_null_i, bins=50, lw=2, color=utils.dna_rna_color_dict['ratio'], histtype='step', density=True, alpha=0.8, zorder=1, label='Null')
    ax_sine_null.axvline(x=slope, lw=3, ls='--', c='k', zorder=2, label='Observed')
    #ax.set_xlim([-0.55,0.55])
    ax_sine_null.set_title(focal_otu_formatted + '\nNull: Gamma with oscillating ' + r'$K_{i}(t)$', fontsize=11)
    ax_sine_null.legend(loc='upper right')
    #ax_sine_null.set_xlabel("Slope between RNA - DNA at time " + r'$t$' + '\nand change in DNA', fontsize=10)
    ax_sine_null.set_xlabel("Slope between change in DNA and change in RNA", fontsize=10)
    ax_sine_null.set_ylabel("Probability density", fontsize=10)



    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig_name = "%sfig4.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



    # histogram....
    fig = plt.figure(figsize = (4.5, 4)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])

    ax.hist(slope_null_time_all, bins=50, lw=2, color=utils.dna_rna_color_dict['ratio'], histtype='step', density=True, alpha=0.8, zorder=1, label='Null from time label permutation')
    ax.axvline(x=slope, lw=3, ls='--', c='k', zorder=2, label='Observed')

    ax.set_xlabel("Slope between change in DNA and change in RNA", fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    ax.legend(loc='upper left')


    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig_name = "%stime_perm_hist.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_slope_delta_null(make_dict=False, n_iter=10000):

    if make_dict == True:

        slope_delta_null_dict = {}
        slope_delta_null_dict['rho_interval'] = {}

        rhogamma_all = numpy.linspace(0, 0.95, endpoint=True, num=20)

        for rhogamma in rhogamma_all:

            slope_null_i, rho_null_i = sine_slope_delta_null(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=rhogamma, n_iter=n_iter)
            #slope_null_i.sort()
            rho_null_i.sort()
            
            # CIs of the correlation
            lower_ci = rho_null_i[int(0.025*n_iter)]
            upper_ci = rho_null_i[int(0.975*n_iter)]

            slope_delta_null_dict['rho_interval'][rhogamma] = {}
            slope_delta_null_dict['rho_interval'][rhogamma]['lower_ci'] = lower_ci
            slope_delta_null_dict['rho_interval'][rhogamma]['upper_ci'] = upper_ci

            print(rhogamma, lower_ci, upper_ci)


        rho_uniform_all = numpy.random.uniform(low=0.0, high=0.95, size=n_iter)
        slope_inferred_all = []
        rho_inferred_all = []
        for rho_uniform in rho_uniform_all:
            slope_j, rho_j = sine_slope_delta_null_one_iter(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=rho_uniform)
            slope_inferred_all.append(slope_j)
            rho_inferred_all.append(rho_j)

        slope_delta_null_dict['rho_uniform'] = rho_uniform_all
        slope_delta_null_dict['rho_inferred_all'] = rho_inferred_all

        sys.stderr.write("Saving dictionary...\n")
        with open(slope_delta_null_dict_path, 'wb') as outfile:
            pickle.dump(slope_delta_null_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stderr.write("Done!\n")


    # load dictionary and plot.
    slope_delta_null_dict = pickle.load(open(slope_delta_null_dict_path, "rb"))

    diff_clr_afd_dna = (clr_afd_dna[1:] - clr_afd_dna[:-1])/(days[1:] - days[:-1])
    diff_clr_afd_rna = (clr_afd_rna[1:] - clr_afd_rna[:-1])/(days[1:] - days[:-1])

    fig = plt.figure(figsize = (8.5, 4)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_rho = fig.add_subplot(gs[0, 1])

    ax_scatter.text(-0.095, 1.06, utils.sub_plot_labels[0], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_scatter.transAxes)
    ax_rho.text(-0.095, 1.06, utils.sub_plot_labels[1], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_rho.transAxes)


    ax_scatter.scatter(diff_clr_afd_dna, diff_clr_afd_rna, s=8, alpha=1, c='k', zorder=2)
    ax_scatter.set_xlabel("Per-day change in DNA, " + r'$\frac{\delta c^{\mathrm{DNA}}}{\delta t}$', fontsize=10)
    ax_scatter.set_ylabel("per-day change in RNA, " + r'$\frac{\delta c^{\mathrm{RNA}}}{\delta t}$', fontsize=10)
    ax_scatter.set_title( focal_otu_formatted + '\nNull: Time label permutation', fontsize=11)

    slope, intercept, r_value, p_value, std_err = stats.linregress(diff_clr_afd_dna, diff_clr_afd_rna)
    print(r_value, p_value)

    # optimal rhogamma
    rho_uniform = numpy.asarray(slope_delta_null_dict['rho_uniform'])
    rho_inferred_all = numpy.asarray(slope_delta_null_dict['rho_inferred_all'])
    dist_rho_uniform_all = numpy.sqrt( (rho_inferred_all - r_value)**2 )
    print(rho_uniform[numpy.argmin(dist_rho_uniform_all)])

    x_range_ =  numpy.linspace(min(diff_clr_afd_dna), max(diff_clr_afd_dna), 10000)
    y_fit_range = slope*x_range_ + intercept
    ax_scatter.plot(x_range_, y_fit_range, ls='--', lw=2.5, c='k')
    ax_scatter.text(0.26, 0.87, r'$\rho = $'  + str(round(r_value, 3)), fontsize=12, ha='center', va='center', transform=ax_scatter.transAxes)
    ax_scatter.text(0.26, 0.78, r'$P < 10^{-20}$', fontsize=12, ha='center', va='center', transform=ax_scatter.transAxes)

    # plot rho range
    rhogamma_all = list(slope_delta_null_dict['rho_interval'].keys())
    rhogamma_all.sort()

    lower_ci_all = [slope_delta_null_dict['rho_interval'][r]['lower_ci'] for r in rhogamma_all]
    upper_ci_all = [slope_delta_null_dict['rho_interval'][r]['upper_ci'] for r in rhogamma_all]

    ax_rho.axhline(y=slope, ls='--', lw=2, c='k', label='Data')
    ax_rho.fill_between(rhogamma_all, lower_ci_all,upper_ci_all, alpha=0.2, color='dodgerblue', label='Simulation, 95% CIs')

    ax_rho.set_xlabel("True correlation b/w " + r'$\frac{\delta c^{\mathrm{DNA}}}{\delta t}$'+ ' and ' + r'$\frac{\delta c^{\mathrm{RNA}}}{\delta t}$', fontsize=10)
    ax_rho.set_ylabel("Inferred correlation b/w " + r'$\frac{\delta c^{\mathrm{DNA}}}{\delta t}$'+ ' and ' + r'$\frac{\delta c^{\mathrm{RNA}}}{\delta t}$', fontsize=10)

    ax_rho.set_xlim([min(rhogamma_all), max(rhogamma_all)])
    ax_rho.plot([min(rhogamma_all), max(rhogamma_all)], [min(rhogamma_all), max(rhogamma_all)], lw=2, ls=':', c='k', label='1:1')

    ax_rho.legend(loc='upper left', fontsize=10)

    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    fig_name = "%sslope_delta_null.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_ratio_vs_delta_dna(rhogamma=0.7418095701816444):

    slope_delta_null_dict = pickle.load(open(slope_delta_null_dict_path, "rb"))
    
    diff_dna_rna = (clr_afd_rna - clr_afd_dna)[:-1]
    delta_dna = (clr_afd_dna[1:] - clr_afd_dna[:-1])/(days[1:] - days[:-1])


    fig = plt.figure(figsize = (8.5, 4)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=1, ncols=2)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])


    ax_scatter.set_xlabel("RNA:DNA at time " + r'$t$' + ', ' + r'$\phi_{\mathrm{photo}}(t)$', fontsize=10)
    ax_scatter.set_ylabel("Per-day change in DNA, " + r'$\delta c_{\mathrm{photo}}^{\mathrm{DNA}} / \delta t $', fontsize=10)
    ax_scatter.scatter(diff_dna_rna, delta_dna, s=8, alpha=1, c='k', zorder=2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(diff_dna_rna, delta_dna)

    x_range_ =  numpy.linspace(min(diff_dna_rna), max(diff_dna_rna), 10000)
    y_fit_range = slope*x_range_ + intercept
    ax_scatter.plot(x_range_, y_fit_range, ls='--', lw=2.5, c='k')
    

    x_range_ci, y_range_pred, lcb, ucb = utils.get_confidence_hull(diff_dna_rna, delta_dna)
    idx_to_plot = (x_range_ci >= min(diff_dna_rna)) & (x_range_ci <= max(diff_dna_rna))
    ax_scatter.plot(x_range_ci[idx_to_plot], lcb[idx_to_plot], color='k', linestyle=':', linewidth=2, zorder=3, label=r'$95\%$' + ' confidence hull')
    ax_scatter.plot(x_range_ci[idx_to_plot], ucb[idx_to_plot], color='k', linestyle=':', linewidth=2, zorder=3)

    ax_scatter.legend(loc='upper left',fontsize=9)
    ax_scatter.text(0.26, 0.81, r'$\rho = $' + str(round(r_value, 3)), fontsize=11, ha='center', va='center', transform=ax_scatter.transAxes)
    ax_scatter.text(0.26, 0.72, r'$P = $' + str(round(p_value, 5)), fontsize=11, ha='center', va='center', transform=ax_scatter.transAxes)
    ax_scatter.set_title(focal_otu_formatted, fontsize=11)

    # histogram
    # optimal slope 
    sine_slope_null, sine_r_value_null = sine_slope_ratio_vs_dna_w_corr_null(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=rhogamma)
    sine_slope_null_0, sine_r_value_null_0 = sine_slope_ratio_vs_dna_w_corr_null(k_dna, k_rna, sigma_dna, sigma_rna, rhogamma=0)

    p_value = sum(sine_slope_null > slope)/len(sine_slope_null)
    p_value_0 = sum(sine_slope_null_0 > slope)/len(sine_slope_null_0)

    p_value_r = sum(sine_r_value_null > r_value)/len(sine_r_value_null)
    p_value_r_0 = sum(sine_r_value_null_0 > r_value)/len(sine_r_value_null_0)

    print(p_value_r, p_value_r_0)

    ax_hist.hist(sine_r_value_null, bins=50, lw=2, color=utils.dna_rna_color_dict['RNA'], histtype='step', density=True, alpha=1, zorder=1, label='Correlated RNA and DNA')
    ax_hist.hist(sine_r_value_null_0, bins=50, ls=':', lw=2, color=utils.dna_rna_color_dict['ratio'], histtype='step', density=True, alpha=1, zorder=1, label='Independent RNA and DNA')

    ax_hist.axvline(x=r_value, lw=3, ls='--', c='k', zorder=2, label='Observed')
    ax_hist.set_title('Null: gamma with oscillating ' + r'$K_{\mathrm{photo}}(t)$', fontsize=11)
    ax_hist.legend(loc='upper left', fontsize=9)
    ax_hist.set_xlabel("Correlation between " + r'$\phi_{\mathrm{photo}}(t)$' + ' and ' + r'$\frac{\delta c_{\mathrm{photo}}^{\mathrm{DNA}}}{\delta t}$', fontsize=10)
    ax_hist.set_ylabel("Probability density", fontsize=10)


    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    fig_name = "%sratio_vs_delta_dna.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")


    #sine_slope_ratio_vs_dna_w_corr_null()
    #plot_ratio_vs_dna()

    #plot_slope_delta_null(make_dict=False)

    plot_ratio_vs_delta_dna()
