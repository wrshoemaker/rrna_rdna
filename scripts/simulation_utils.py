import os
import sys
import pickle
import numpy
from scipy.special import polygamma, digamma
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import gmean
from scipy.optimize import fsolve


import utils
import config
import sine_parameter_utils
from lmfit import Minimizer, create_params, fit_report
import itertools

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

numpy.random.seed(123456789)


param_oscillation_artifact_simulation_path = config.data_directory + 'param_oscillation_artifact_simulation_dict.pickle'
param_oscillation_artifact_simulation_clr_all_otus_path = config.data_directory + 'param_oscillation_artifact_simulation_clr_all_otus_dict.pickle'


compare_clr_to_true_abundance_dict_path = config.data_directory + 'compare_clr_to_true_abundance_dict.pickle'
compare_sigma_clr_to_true_abundance_oscillating_dict_path = config.data_directory + 'compare_sigma_clr_to_true_abundance_oscillating_dict.pickle'
data_collapse_simulation_path = config.data_directory + 'data_collapse_simulation.pickle'


method_label_dict = {'log_rel': 'Rescaled log rel.', 'clr': 'CLR'}


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])


gm_color = {0.1:'lightskyblue', 0.3:'dodgerblue', 0.5:'royalblue'}


gm_color_clr = {0.1:'lightskyblue', 0.3:'dodgerblue', 0.5:'royalblue'}
gm_color_rel = {0.1:'coral', 0.3:'orangered', 0.5:'firebrick'}

otu_type_all = ['focal', 'nonfocal']

#amp_colormap = utils.make_colormap('DNA', len(amp_focal_range))



def generate_community_from_sigma_k(S, k, sigma, n_sites, N, rhogamma=0):

    # Extraction of the who vectors of abundances, distributed according to Gamma distributions with the correlation rhogamma
    cov = numpy.ones((n_sites, n_sites))
    I = numpy.identity(n_sites)
    cov = ((cov-I)*rhogamma) + I

    Z = numpy.random.multivariate_normal(numpy.asarray([0]*n_sites), cov, S, tol=1e-5)
    U = norm.cdf(Z)

    # check if the carrying capacity for each species is a vector (i.e., time-dependent) or a constant
    if len(k.shape) == 1:
        abundances_all = [gamma.ppf(U[:,idx], numpy.divide(2,sigma)-1, scale=sigma*k[idx]/2) for idx in range(n_sites)]

    else:
        # a shape of 2D, representing a matrix
        abundances_all = [gamma.ppf(U[:,idx], numpy.divide(2,sigma)-1, scale=sigma*k[idx,:]/2) for idx in range(n_sites)]
    
    #abundances_all = [gamma.ppf(U[:,idx], numpy.divide(2,sigma)-1, scale=sigma*k[idx,:]/2) for idx in range(n_sites)]
    abundances_all = numpy.asarray(abundances_all).T
    # Normalise, to have relative abundances
    rel_abundances_all =  abundances_all/numpy.sum(abundances_all, axis=0)

    # run multinomial
    read_counts_multinomial_all = []
    #read_counts_poisson_all = []
    for sad_idx, sad in enumerate(rel_abundances_all.T):
        read_counts_multinomial_all.append(numpy.random.multinomial(int(N[sad_idx]), sad))

    read_counts_multinomial_all = numpy.asarray(read_counts_multinomial_all).T

    # remove undetected species
    non_zero_idx = numpy.sum(read_counts_multinomial_all, axis=1) > 0
    read_counts_multinomial_all_nonzero = read_counts_multinomial_all[non_zero_idx,:]

    return abundances_all, rel_abundances_all, read_counts_multinomial_all, read_counts_multinomial_all_nonzero, non_zero_idx




def generate_community(mu, s, S, N, dist, gm, n_sites, rhogamma=0):
    #This function  generates the parameters K and sigma for two communities with given parameters (mu, s),
    # S species, sigma^2 distributed as 'dist' (if dist is exponential, with average gm), correlation rho between the values of K
    # and correlation rhogamma between the Gamma-distributed fluctuations of abundance

    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    K = numpy.exp(numpy.random.normal(mu, s, S)) #correlated K for the two communities extracted from lognormal dist
    sigmarnd=[]  #Exponentially distributed sigma, common for the two communities
    if dist=='exp':
        for k in range(S):
            tr=100
            while tr>1.95: # Values too close to 2 give numerical problems when extracting from the Gamma distribution
                tr=numpy.sqrt(numpy.random.exponential(gm))

            sigmarnd.append(tr)
    else:
        if dist=='unif':
            sigmarnd=numpy.random.uniform(0,1.95,size=S)

        if dist == 'constant':
            sigmarnd=numpy.repeat(gm, S)


    # Extraction of the who vectors of abundances, distributed according to Gamma distributions with the correlation rhogamma
    cov = numpy.ones((n_sites, n_sites))
    I = numpy.identity(n_sites)
    cov = ((cov-I)*rhogamma) + I

    Z = numpy.random.multivariate_normal(numpy.asarray([0]*n_sites), cov, S, tol=1e-5)
    U = norm.cdf(Z)

    abundances_all = []
    for idx in range(n_sites):
        G = gamma.ppf(U[:,idx], numpy.divide(2,sigmarnd)-1, scale=sigmarnd*K/2)
        abundances_all.append(G)

    abundances_all = numpy.asarray(abundances_all).T
    # Normalise, to have relative abundances
    rel_abundances_all =  abundances_all/numpy.sum(abundances_all, axis=0)

    # run multinomial
    read_counts_multinomial_all = []
    #read_counts_poisson_all = []
    for sad_idx, sad in enumerate(rel_abundances_all.T):
        read_counts_multinomial_all.append(numpy.random.multinomial(int(N[sad_idx]), sad))
        #read_counts_poisson_all.append(numpy.random.poisson(lam=int(N[sad_idx])*rel_abundances_all))

    read_counts_multinomial_all = numpy.asarray(read_counts_multinomial_all).T


    return rel_abundances_all, read_counts_multinomial_all



def generate_community_oscillating_k(mu, s, S, N, dist, gm, n_sites, amp_focal=4, rhogamma=0):

    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    # carrying capacity can be interpreted 
    # the logarithm of carrying capacities follow a sine wave
    log_K_0 = numpy.random.normal(mu, s, S)
    # sort so that carrying capacities are increasing
    log_K_0 = numpy.sort(log_K_0)

    amp = numpy.repeat(0, repeats=S-1)
    amp = numpy.append(amp, amp_focal)
    
    freq = numpy.repeat(1, repeats=S-1)
    freq = numpy.append(freq, 0.018)

    phase = numpy.repeat(0, repeats=S-1)
    phase = numpy.append(phase, 1.8)

    log_K_t = (numpy.sin(numpy.outer(days, freq) + phase) * amp) + log_K_0
    K_t = numpy.exp(log_K_t)

    sigmarnd = []  #Exponentially distributed sigma, common for the two communities
    if dist == 'exp':
        for k in range(S):
            tr = 100
            while tr > 1.95: # Values too close to 2 give numerical problems when extracting from the Gamma distribution
                tr = numpy.sqrt(numpy.random.exponential(gm))

            sigmarnd.append(tr)
    else:
        if dist == 'unif':
            sigmarnd = numpy.random.uniform(0, 1.95, size=S)

        if dist == 'constant':
            sigmarnd = numpy.repeat(gm, S)

    # Extraction of the who vectors of abundances, distributed according to Gamma distributions with the correlation rhogamma
    cov = numpy.ones((n_sites, n_sites))
    I = numpy.identity(n_sites)
    cov = ((cov-I)*rhogamma) + I

    Z = numpy.random.multivariate_normal(numpy.asarray([0]*n_sites), cov, S, tol=1e-5)
    U = norm.cdf(Z)

    abundances_all = [gamma.ppf(U[:,idx], numpy.divide(2,sigmarnd)-1, scale=sigmarnd*K_t[idx,:]/2) for idx in range(n_sites)]
    abundances_all = numpy.asarray(abundances_all).T
    # Normalise, to have relative abundances
    rel_abundances_all =  abundances_all/numpy.sum(abundances_all, axis=0)

    # run multinomial
    read_counts_multinomial_all = []
    #read_counts_poisson_all = []
    for sad_idx, sad in enumerate(rel_abundances_all.T):
        read_counts_multinomial_all.append(numpy.random.multinomial(int(N[sad_idx]), sad))

    read_counts_multinomial_all = numpy.asarray(read_counts_multinomial_all).T

    return rel_abundances_all, read_counts_multinomial_all



def test_amp_effect_fix_mean_var(mu, s, S, N, dist, gm, n_sites, rhogamma=0):

    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    # carrying capacity can be interpreted 
    # the logarithm of carrying capacities follow a sine wave
    log_K_0 = numpy.random.normal(mu, s, S)
    # sort so that carrying capacities are increasing
    log_K_0 = numpy.sort(log_K_0)
    
    freq = numpy.repeat(1, repeats=S-1)
    freq = numpy.append(freq, 0.018)

    phase = numpy.repeat(0, repeats=S-1)
    phase = numpy.append(phase, 1.8)

    sigmarnd = []  #Exponentially distributed sigma, common for the two communities
    if dist == 'exp':
        for k in range(S):
            tr = 100
            while tr > 1.95: # Values too close to 2 give numerical problems when extracting from the Gamma distribution
                tr = numpy.sqrt(numpy.random.exponential(gm))

            sigmarnd.append(tr)
    else:
        if dist == 'unif':
            sigmarnd = numpy.random.uniform(0, 1.95, size=S)

        if dist == 'constant':
            sigmarnd = numpy.repeat(gm, S)

    # Extraction of the who vectors of abundances, distributed according to Gamma distributions with the correlation rhogamma
    cov = numpy.ones((n_sites, n_sites))
    I = numpy.identity(n_sites)
    cov = ((cov-I)*rhogamma) + I

    Z = numpy.random.multivariate_normal(numpy.asarray([0]*n_sites), cov, S, tol=1e-5)
    U = norm.cdf(Z)
    
    n_amp_focal = 5
    #amp_focal_range = numpy.logspace(numpy.log10(0.01), numpy.log10(20), base=10, num=n_amp_focal, endpoint=True)
    amp_focal_range = [0, 4, 8, 12, 16]
    amp_colormap = utils.make_colormap('DNA', len(amp_focal_range))
    #print(amp_focal_range)

    focal_sine_dict = {}
    #s_by_s_sampled_all = []
    fig = plt.figure(figsize = (12, 8))
    ax_focal = plt.subplot2grid((2, 3), (0, 0))
    ax_rank_2 = plt.subplot2grid((2, 3), (0, 1))
    ax_rank_2_no_focal = plt.subplot2grid((2, 3), (0, 2))

    ax_focal_reads = plt.subplot2grid((2, 3), (1, 0))
    ax_rank_2_reads = plt.subplot2grid((2, 3), (1, 1))
    ax_rank_2_no_focal_reads = plt.subplot2grid((2, 3), (1, 2))

    for amp_focal_idx, amp_focal in enumerate(amp_focal_range):

        print(amp_focal, n_sites)

        amp = numpy.repeat(0, repeats=S-1)
        amp = numpy.append(amp, amp_focal)

        log_K_t = (numpy.sin(numpy.outer(days, freq) + phase) * amp) + log_K_0
        K_t = numpy.exp(log_K_t)

        abundances_all = [gamma.ppf(U[:,idx], numpy.divide(2,sigmarnd)-1, scale=sigmarnd*K_t[idx,:]/2) for idx in range(n_sites)]
        abundances_all = numpy.asarray(abundances_all).T
        # Normalise, to have relative abundances
        rel_abundances_all =  abundances_all/numpy.sum(abundances_all, axis=0)

        # run multinomial
        read_counts_multinomial_all = []
        #read_counts_poisson_all = []
        for sad_idx, sad in enumerate(rel_abundances_all.T):
            read_counts_multinomial_all.append(numpy.random.multinomial(int(N[sad_idx]), sad))

        read_counts_multinomial_all = numpy.asarray(read_counts_multinomial_all).T

        #utils.clr_transform(rel_read_counts_multinomial_all)

        rel_read_counts_multinomial_all = read_counts_multinomial_all/numpy.sum(read_counts_multinomial_all, axis=0)
        #s_by_s_sampled_all.append(rel_read_counts_multinomial_all)

        focal_afd = rel_read_counts_multinomial_all[-1,:]
        # rescale like in the data.
        focal_afd = focal_afd/numpy.mean(focal_afd)
        to_plot_focal_idx = focal_afd>0
        focal_afd_to_plot = focal_afd[to_plot_focal_idx]
        focal_days_to_plot = days[to_plot_focal_idx]
        ax_focal.plot(focal_days_to_plot, focal_afd_to_plot, c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)

        # reads
        focal_afd_reads = read_counts_multinomial_all[-1,:]
        to_plot_focal_reads_idx = focal_afd_reads>0
        focal_afd_reads_to_plot = focal_afd_reads[to_plot_focal_reads_idx]
        focal_reads_days_to_plot = days[to_plot_focal_reads_idx]
        ax_focal_reads.plot(focal_reads_days_to_plot, focal_afd_reads_to_plot, c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)


        rank_2_afd = rel_read_counts_multinomial_all[-2,:]
        rank_2_afd = rank_2_afd/numpy.mean(rank_2_afd)
        to_plot_rank_2_idx = rank_2_afd>0
        rank_2_afd_to_plot = rank_2_afd[to_plot_rank_2_idx]
        rank_2_days_to_plot = days[to_plot_rank_2_idx]
        ax_rank_2.plot(rank_2_days_to_plot, rank_2_afd_to_plot, c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)

        # reads
        rank_2_afd_reads = read_counts_multinomial_all[-2,:]
        to_plot_rank_2_reads_idx = rank_2_afd_reads>0
        rank_2_afd_reads_to_plot = rank_2_afd_reads[to_plot_rank_2_reads_idx]
        rank_2_days_reads_to_plot = days[to_plot_rank_2_reads_idx]
        ax_rank_2_reads.plot(rank_2_days_reads_to_plot, rank_2_afd_reads_to_plot, c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)


        # plot without focal
        read_counts_multinomial_all_no_focal = read_counts_multinomial_all[:-1,:]
        # remove zeros
        n_reads = numpy.sum(read_counts_multinomial_all_no_focal, axis=0)
        samples_to_keep_idx = n_reads > 0 
        days_to_keep = days[samples_to_keep_idx]
        read_counts_multinomial_all_no_focal = read_counts_multinomial_all_no_focal[:,samples_to_keep_idx]
        rel_read_counts_multinomial_all_no_focal = read_counts_multinomial_all_no_focal/numpy.sum(read_counts_multinomial_all_no_focal, axis=0)
        rank_2_afd_no_focal = rel_read_counts_multinomial_all_no_focal[-1,:]
        rank_2_afd_no_focal = rank_2_afd_no_focal/numpy.mean(rank_2_afd_no_focal)
        to_plot_rank_2_no_focal_idx = rank_2_afd_no_focal>0
        rank_2_afd_no_focal_to_plot = rank_2_afd_no_focal[to_plot_rank_2_no_focal_idx]
        rank_2_days_no_focal_to_plot = days_to_keep[to_plot_rank_2_no_focal_idx]
        ax_rank_2_no_focal.plot(rank_2_days_no_focal_to_plot, rank_2_afd_no_focal_to_plot, c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)


        ax_rank_2_no_focal_reads.plot(days[samples_to_keep_idx], n_reads[samples_to_keep_idx], c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)



    ax_focal.set_yscale('log', basey=10)
    ax_focal.set_xlabel("Days", fontsize=10)
    ax_focal.set_ylabel("Rescaled relative abundance", fontsize=10)
    ax_focal.legend(loc='lower right', fontsize=8)
    ax_focal.set_title('Oscillating OTU', fontsize=12)

    ax_focal_reads.set_yscale('log', basey=10)
    ax_focal_reads.set_xlabel("Days", fontsize=10)
    ax_focal_reads.set_ylabel("Number of reads", fontsize=10)
    ax_focal_reads.set_title('Oscillating OTU', fontsize=12)

    ax_rank_2.set_yscale('log', basey=10)
    ax_rank_2.set_xlabel("Days", fontsize=10)
    ax_rank_2.set_ylabel("Rescaled relative abundance", fontsize=10)
    ax_rank_2.set_title('Non-oscillating OTU, rank 2 abundance', fontsize=12)

    ax_rank_2_reads.set_yscale('log', basey=10)
    ax_rank_2_reads.set_xlabel("Days", fontsize=10)
    ax_rank_2_reads.set_ylabel("Number of reads", fontsize=10)
    ax_rank_2_reads.set_title('Non-oscillating OTU, rank 2 abundance', fontsize=12)

    #ax_rank_2_no_focal.set_ylim([0.5e-4, 7])
    ax_rank_2_no_focal.set_yscale('log', basey=10)
    ax_rank_2_no_focal.set_xlabel("Days", fontsize=10)
    ax_rank_2_no_focal.set_ylabel("Rescaled relative abundance", fontsize=10)
    ax_rank_2_no_focal.set_title('Non-oscillating OTU, rank 2 abundance\nOscillating OTU removed from reads', fontsize=12)

    # total number reads
    ax_rank_2_no_focal_reads.set_yscale('log', basey=10)
    ax_rank_2_no_focal_reads.set_xlabel("Days", fontsize=10)
    ax_rank_2_no_focal_reads.set_ylabel("Total number of reads", fontsize=10)
    ax_rank_2_no_focal_reads.set_title('Oscillating OTU removed from reads', fontsize=12)

    fig.subplots_adjust(hspace=0.35,wspace=0.4)
    fig_name = "%stest_amp_effect_fix_mean_var.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def test_amp_effect_fix_mean_var_clr(mu, s, S, N, dist, gm, n_sites, rhogamma=0):

    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    # carrying capacity can be interpreted 
    # the logarithm of carrying capacities follow a sine wave
    log_K_0 = numpy.random.normal(mu, s, S)
    # sort so that carrying capacities are increasing
    log_K_0 = numpy.sort(log_K_0)
    
    freq = numpy.repeat(1, repeats=S-1)
    freq = numpy.append(freq, 0.018)

    phase = numpy.repeat(0, repeats=S-1)
    phase = numpy.append(phase, 1.8)

    sigmarnd = []  #Exponentially distributed sigma, common for the two communities
    if dist == 'exp':
        for k in range(S):
            tr = 100
            while tr > 1.95: # Values too close to 2 give numerical problems when extracting from the Gamma distribution
                tr = numpy.sqrt(numpy.random.exponential(gm))

            sigmarnd.append(tr)
    else:
        if dist == 'unif':
            sigmarnd = numpy.random.uniform(0, 1.95, size=S)

        if dist == 'constant':
            sigmarnd = numpy.repeat(gm, S)

    # Extraction of the who vectors of abundances, distributed according to Gamma distributions with the correlation rhogamma
    cov = numpy.ones((n_sites, n_sites))
    I = numpy.identity(n_sites)
    cov = ((cov-I)*rhogamma) + I

    Z = numpy.random.multivariate_normal(numpy.asarray([0]*n_sites), cov, S, tol=1e-5)
    U = norm.cdf(Z)
    
    n_amp_focal = 5
    #amp_focal_range = numpy.logspace(numpy.log10(0.01), numpy.log10(20), base=10, num=n_amp_focal, endpoint=True)
    amp_focal_range = [0, 2, 4, 6, 8]
    amp_colormap = utils.make_colormap('DNA', len(amp_focal_range))
    #print(amp_focal_range)

    focal_sine_dict = {}
    #s_by_s_sampled_all = []
    fig = plt.figure(figsize = (12, 8))
    ax_focal = plt.subplot2grid((2, 3), (0, 0))
    ax_rank_2 = plt.subplot2grid((2, 3), (0, 1))
    ax_rank_2_no_focal = plt.subplot2grid((2, 3), (0, 2))



    for amp_focal_idx, amp_focal in enumerate(amp_focal_range):

        amp = numpy.repeat(0, repeats=S-1)
        amp = numpy.append(amp, amp_focal)

        log_K_t = (numpy.sin(numpy.outer(days, freq) + phase) * amp) + log_K_0
        K_t = numpy.exp(log_K_t)

        abundances_all = [gamma.ppf(U[:,idx], numpy.divide(2,sigmarnd)-1, scale=sigmarnd*K_t[idx,:]/2) for idx in range(n_sites)]
        abundances_all = numpy.asarray(abundances_all).T
        # Normalise, to have relative abundances
        rel_abundances_all =  abundances_all/numpy.sum(abundances_all, axis=0)

        # run multinomial
        read_counts_multinomial_all = []
        #read_counts_poisson_all = []
        for sad_idx, sad in enumerate(rel_abundances_all.T):
            read_counts_multinomial_all.append(numpy.random.multinomial(int(N[sad_idx]), sad))

        read_counts_multinomial_all = numpy.asarray(read_counts_multinomial_all).T

        clr_s_by_s, occupancy_idx = utils.clr_transform_sim(read_counts_multinomial_all)
        
        focal_afd = clr_s_by_s[-1,:]
        # rescale like in the data.
        focal_afd = focal_afd - numpy.mean(focal_afd)
        #to_plot_focal_idx = focal_afd>0
        #focal_afd_to_plot = focal_afd[to_plot_focal_idx]
        #focal_days_to_plot = days[to_plot_focal_idx]
        ax_focal.plot(days, focal_afd, c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)


        rank_2_afd = clr_s_by_s[-2,:]
        rank_2_afd = rank_2_afd - numpy.mean(rank_2_afd)
        #print(rank_2_afd)
        #to_plot_rank_2_idx = rank_2_afd>0
        #rank_2_afd_to_plot = rank_2_afd[to_plot_rank_2_idx]
        #rank_2_days_to_plot = days[to_plot_rank_2_idx]
        ax_rank_2.plot(days, rank_2_afd, c=amp_colormap[amp_focal_idx], lw=2, alpha=0.8, label='Amp = %0.2f' % amp_focal)

        

    ax_focal.set_xlabel("Days", fontsize=10)
    ax_focal.set_ylabel("Rescaled relative abundance", fontsize=10)
    ax_focal.legend(loc='lower right', fontsize=8)
    ax_focal.set_title('Oscillating OTU', fontsize=12)


    ax_rank_2.set_xlabel("Days", fontsize=10)
    ax_rank_2.set_ylabel("Rescaled relative abundance", fontsize=10)
    ax_rank_2.set_title('Non-oscillating OTU, rank 2 abundance', fontsize=12)


    #ax_rank_2_no_focal.set_ylim([0.5e-4, 7])
    ax_rank_2_no_focal.set_yscale('log', basey=10)
    ax_rank_2_no_focal.set_xlabel("Days", fontsize=10)
    ax_rank_2_no_focal.set_ylabel("Rescaled relative abundance", fontsize=10)
    ax_rank_2_no_focal.set_title('Non-oscillating OTU, rank 2 abundance\nOscillating OTU removed from reads', fontsize=12)


    fig.subplots_adjust(hspace=0.35,wspace=0.4)
    fig_name = "%stest_amp_effect_fix_mean_var_clr.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def oscillation_artifact_simulation(mu, s, S, N, dist, gm_all, n_sites, focal_amp_all=[4], focal_freq_all=[2*numpy.pi/365], focal_phase_all=[1.8], n_iter=1, rhogamma=0, clr_all_otus=False):

    # clr_all_otus = boolean, asks whether to use CLR for all OTUs with an added pseudocount of one.

    # Fix distribution of K_0
    # Loop through iteration
    # For each iteration draw sigmas, and abundanaces from distribution using each sine parameter combination
    # Fit model
    # get parameters

    sine_param_combo_all = list(itertools.product(focal_amp_all, focal_freq_all, focal_phase_all))

    param_dict = {}
    param_dict['true_abundance'] = {}
    param_dict['clr'] = {}
    param_dict['log_rel'] = {}

    for otu_type in otu_type_all:
        param_dict['true_abundance'][otu_type] = {}
        param_dict['clr'][otu_type] = {}
        param_dict['log_rel'][otu_type] = {}
    
        for gm in gm_all:
            param_dict['true_abundance'][otu_type][gm] = {}
            param_dict['clr'][otu_type][gm] = {}
            param_dict['log_rel'][otu_type][gm] = {}

            for sine_param_combo in sine_param_combo_all:
                param_dict['true_abundance'][otu_type][gm][sine_param_combo] = {}
                param_dict['clr'][otu_type][gm][sine_param_combo] = {}
                param_dict['log_rel'][otu_type][gm][sine_param_combo] = {}

                param_dict['clr'][otu_type][gm][sine_param_combo]['num_sampled_species'] = []
                param_dict['log_rel'][otu_type][gm][sine_param_combo]['num_sampled_species'] = []

                param_dict['true_abundance'][otu_type][gm][sine_param_combo]['afd'] = []
                param_dict['clr'][otu_type][gm][sine_param_combo]['afd'] = []
                param_dict['log_rel'][otu_type][gm][sine_param_combo]['afd'] = []

                for p in sine_parameter_utils.param_no_method_all:
                    param_dict['clr'][otu_type][gm][sine_param_combo]['%s_mle' % p] = []
                    param_dict['log_rel'][otu_type][gm][sine_param_combo]['%s_mle' % p] = []


    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    # carrying capacity can be interpreted 
    # the logarithm of carrying capacities follow a sine wave
    log_K_0 = numpy.random.normal(mu, s, S)
    # sort so that carrying capacities are increasing
    log_K_0 = numpy.sort(log_K_0)

    sys.stderr.write("Running simulations...\n")
    #for n in range(n_iter):
    while len(param_dict['clr'][otu_type][gm][sine_param_combo]['amp_mle']) < n_iter:

        skip_iter = False

        # we want all iterations to have the same sample of the sigma distribution
        # chack for all gm and all sine parameter combinations whether you get AFDs
        # where rank one and rank two have no zeros
        afd_iter_dict = {}
        # fix carrying capacity
        for gm in gm_all:

            afd_iter_dict[gm] = {}

            sigmarnd = []  #Exponentially distributed sigma, common for the two communities
            if dist == 'exp':
                for k in range(S):
                    tr = 100
                    while tr > 1.95: # Values too close to 2 give numerical problems when extracting from the Gamma distribution
                        tr = numpy.sqrt(numpy.random.exponential(gm))
                    sigmarnd.append(tr)

                sigmarnd = numpy.asarray(sigmarnd)
                # sigma defined on range 0 < sigma < 2


            else:
                if dist == 'unif':
                    sigmarnd = numpy.random.uniform(0, 1.95, size=S)

                if dist == 'constant':
                    sigmarnd = numpy.repeat(gm, S)
            
            # loop through the oscillation parameters
            # keep K_0 and sigma FIXED for all simulations.
            for sine_param_combo in sine_param_combo_all:

                focal_amp, focal_freq, focal_phase = sine_param_combo[0], sine_param_combo[1], sine_param_combo[2]

                amp = numpy.repeat(0, repeats=S-1)
                amp = numpy.append(amp, focal_amp)

                freq = numpy.repeat(1, repeats=S-1)
                freq = numpy.append(freq, focal_freq)

                phase = numpy.repeat(0, repeats=S-1)
                phase = numpy.append(phase, focal_phase)

                # generate carrying capacity over time.
                K_t = numpy.exp((numpy.sin(numpy.outer(days, freq) + phase) * amp) + log_K_0)

                abundances_all, rel_abundances_all, read_counts_multinomial_all, read_counts_multinomial_all_nonzero, non_zero_idx = generate_community_from_sigma_k(S, K_t, sigmarnd, n_sites, N)

                # relative abundance
                rel_read_counts_multinomial_all_nonzero = read_counts_multinomial_all_nonzero/numpy.sum(read_counts_multinomial_all_nonzero, axis=0)
                # all species are used to calculate relative abundance

                # CLR
                if clr_all_otus == True:
                    clr_s_by_s, occupancy_idx = utils.clr_transform_sim(read_counts_multinomial_all_nonzero, min_occupancy=1)
                else:
                    clr_s_by_s, occupancy_idx = utils.clr_transform_sim_subset(read_counts_multinomial_all_nonzero, min_occupancy=1)
                


                rescaled_clr_s_by_s = (clr_s_by_s - numpy.mean(clr_s_by_s, axis=0))

                afd_iter_dict[gm][sine_param_combo] = {}
                afd_iter_dict[gm][sine_param_combo]['num_sampled_species_log_rel'] = read_counts_multinomial_all_nonzero.shape[0]
                afd_iter_dict[gm][sine_param_combo]['num_sampled_species_clr'] = sum(occupancy_idx)

                for otu_type in otu_type_all:

                    afd_iter_dict[gm][sine_param_combo][otu_type] = {}

                    if otu_type == 'focal':
                        rank_idx = -1                        
                    else:
                        rank_idx = -2


                    afd_otu = rel_abundances_all[rank_idx,:]
                    afd_otu = abundances_all[rank_idx,:]
                    afd_clr_otu = rescaled_clr_s_by_s[rank_idx,:]
                    afd_log_rel_otu = numpy.log10(rel_read_counts_multinomial_all_nonzero[rank_idx,:])

                    # check for zeros in relative abundance
                    if sum(rel_read_counts_multinomial_all_nonzero[rank_idx,:] == 0) > 0:
                        skip_iter = True
                    
                    if sum(numpy.isnan(afd_clr_otu)) > 0:
                        skip_iter = True
                    
                    afd_iter_dict[gm][sine_param_combo][otu_type]['true_abundance'] = afd_otu
                    afd_iter_dict[gm][sine_param_combo][otu_type]['clr'] = afd_clr_otu
                    afd_iter_dict[gm][sine_param_combo][otu_type]['log_rel'] = afd_log_rel_otu


        # repeat process
        if skip_iter == True:
            continue
        
        # proceed...
        for gm in gm_all:

            for sine_param_combo in sine_param_combo_all:

                focal_amp, focal_freq, focal_phase = sine_param_combo[0], sine_param_combo[1], sine_param_combo[2]
                sys.stderr.write("Parameter sigma exp = %.2f, Amp = %.2f, Freq = %.4f, Phase = %.3f, %s, Iter = %d ...\n" % (gm, focal_amp, focal_freq, focal_phase, otu_type, len(param_dict['clr'][otu_type][gm][sine_param_combo]['amp_mle'])))

                param_dict['log_rel'][otu_type][gm][sine_param_combo]['num_sampled_species'].append(afd_iter_dict[gm][sine_param_combo]['num_sampled_species_log_rel'])
                param_dict['clr'][otu_type][gm][sine_param_combo]['num_sampled_species'].append(afd_iter_dict[gm][sine_param_combo]['num_sampled_species_clr'])

                for otu_type in otu_type_all:
                    
                    afd_true_abundance_otu = afd_iter_dict[gm][sine_param_combo][otu_type]['true_abundance']
                    afd_clr_otu = afd_iter_dict[gm][sine_param_combo][otu_type]['clr']
                    afd_log_rel_otu = afd_iter_dict[gm][sine_param_combo][otu_type]['log_rel']

                    afd_exp_clr_otu = numpy.exp(afd_clr_otu)
                    afd_rel_otu = numpy.exp(afd_log_rel_otu)

                    freq_value = 2*numpy.pi/365 # 0.01721420632
                    freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
                    freq_max = 2*numpy.pi/180 # 0.034906585 (365-185)

                    phase_value = numpy.pi
                    phase_min = 0
                    phase_max = 2*numpy.pi

                    amp_value = 1
                    amp_min = 1e-3
                    amp_max = 10

                    #param_mean_min_clr = -2
                    #param_mean_max_clr = 2
                    #param_mean_value_clr = numpy.mean(afd_clr_otu)
                    #param_mean_value_log_rel = numpy.mean(afd_log_rel_otu)

                    #param_mean_min_log_rel = -0.5
                    #param_mean_max_log_rel = 0.5

                    param_mean_value_exp_afd_clr = numpy.mean(afd_exp_clr_otu)
                    param_min_value_exp_afd_clr = min(afd_exp_clr_otu)
                    param_max_value_exp_afd_clr = max(afd_exp_clr_otu)

                    param_mean_value_afd_rel = numpy.mean(afd_rel_otu)
                    param_min_value_afd_rel = min(afd_rel_otu)
                    param_max_value_afd_rel = max(afd_rel_otu)

     

                    params_afd_exp_clr = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                                                freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                                phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                                param_mean=dict(value=param_mean_value_exp_afd_clr, min=param_min_value_exp_afd_clr, max=param_max_value_exp_afd_clr))


                    params_afd_rel = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                                                freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                                phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                                param_mean=dict(value=param_mean_value_afd_rel, min=param_min_value_afd_rel, max=param_max_value_afd_rel))


                    beta_estimate_clr, sigma_estimate_clr = mle_sigma(afd_exp_clr_otu)
                    result_brute_clr, fitter_clr = sine_parameter_utils.grid_search_mle_sine_wave(days, afd_exp_clr_otu, params_afd_exp_clr, beta_estimate_clr)
                    #best_params_brute_clr = result_brute_clr.params


                    beta_estimate_rel, sigma_estimaterel = mle_sigma(params_afd_rel)
                    result_brute_rel, fitter_rel = sine_parameter_utils.grid_search_mle_sine_wave(days, afd_rel_otu, params_afd_rel, beta_estimate_rel)
                    #best_params_brute_rel = result_brute_rel.params

                    #result_brute_clr, fitter_clr = sine_parameter_utils.grid_search_sine_wave(days, afd_clr_otu, params_exp_afd_clr)
                    #result_brute_log_rel, fitter_log_rel = sine_parameter_utils.grid_search_sine_wave(days, afd_log_rel_otu, params_afd_rel)

                    #best_result_leastsq_clr = sine_parameter_utils.second_rount_optimization(result_brute_clr, fitter_clr)
                    #best_result_leastsq_log_rel = sine_parameter_utils.second_rount_optimization(result_brute_log_rel, fitter_log_rel)

                    #best_params_leastsq_clr = best_result_leastsq_clr.params
                    #best_params_leastsq_log_rel = best_result_leastsq_log_rel.params

                    best_result_mle_clr = sine_parameter_utils.second_round_optimization_mle(result_brute_clr, fitter_clr, beta_estimate_clr)
                    best_params_mle_clr = best_result_mle_clr.params

                    best_result_mle_rel = sine_parameter_utils.second_round_optimization_mle(result_brute_rel, fitter_rel, beta_estimate_rel)
                    best_params_mle_rel = best_result_mle_rel.params


                    param_dict['true_abundance'][otu_type][gm][sine_param_combo]['afd'].append(afd_true_abundance_otu.tolist())
                    param_dict['clr'][otu_type][gm][sine_param_combo]['afd'].append(afd_clr_otu.tolist())
                    param_dict['log_rel'][otu_type][gm][sine_param_combo]['afd'].append(afd_log_rel_otu.tolist())

                    for p in sine_parameter_utils.param_no_method_all:
                        param_dict['clr'][otu_type][gm][sine_param_combo]['%s_mle' % p].append(best_params_mle_clr[p].value)
                        param_dict['log_rel'][otu_type][gm][sine_param_combo]['%s_mle' % p].append(best_params_mle_rel[p].value)

                        if otu_type == 'nonfocal':

                            if p == 'amp':
                                print('Amp', best_params_mle_clr[p].value, best_params_mle_rel[p].value)



    if clr_all_otus == True:
        path_ = param_oscillation_artifact_simulation_clr_all_otus_path
    else:
        path_ = param_oscillation_artifact_simulation_path
    

    sys.stderr.write("Saving parameter dictionary...\n")
    with open(path_, 'wb') as outfile:
        pickle.dump(param_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")


    #fig = plt.figure(figsize = (12, 8))
    #ax_focal = plt.subplot2grid((1, 2), (0, 0))
    #ax_rank_2 = plt.subplot2grid((1, 2), (0, 1))
    
    #ax_focal.plot(days, rescaled_clr_s_by_s[-2,:])
    #ax_rank_2.plot(days, numpy.log10(rescaled_rel_s_by_s[-2,:]))

    #fig.subplots_adjust(hspace=0.35,wspace=0.4)
    #fig_name = "%stest_oscillation_sim.png" % config.analysis_directory
    #fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    #plt.close()


    #print(rescaled_clr_s_by_s[-1,:])
    #print(numpy.log10(rescaled_rel_s_by_s[-1,:]))




def plot_oscillation_artifact_simulation():

    fig = plt.figure(figsize = (8, 8))

    param_dict = pickle.load(open(param_oscillation_artifact_simulation_path, "rb"))

    for method_idx, method in enumerate(['log_rel', 'clr']):

        for rank_idx, rank in enumerate(['focal', 'nonfocal']):

            ax = plt.subplot2grid((2, 2), (method_idx, rank_idx))

            ax.text(-0.1, 1.07, utils.sub_plot_labels[method_idx+rank_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)


            for gm_idx, gm in enumerate(list(param_dict[method][rank].keys())):

                param_combo_all = list(param_dict[method][rank][gm].keys())

                #amp_first_rank = [s[0] for s in param_combo_all]
                amp_first_rank = [s[0] for s in param_combo_all]

                amp_inferred = [numpy.mean(param_dict[method][rank][gm][p]['amp_mle']) for p in param_combo_all]

                #print([numpy.mean(param_dict[method][rank][gm][p]['freq_leastsq']) for p in param_combo_all])
                if gm_idx == 0:

                    if rank == 'focal':
                        ax.plot([min(amp_first_rank), max(amp_first_rank)], [min(amp_first_rank), max(amp_first_rank)], ls=':', lw=2, c='k', label='1:1')
                        ax.set_ylabel('Inferred amplitude of focal OTU', fontsize=11)
                    else:
                        ax.axhline(y=0, ls=':', lw=2, c='k', label='True amplitude of non-focal OTU')
                        ax.set_ylabel('Inferred amplitude of non-focal OTU', fontsize=11)


                    if method == 'log_rel':
                        ax.set_title('Relative abundance', fontsize=12)
                    else:
                        ax.set_title('CLR-transformed abundance', fontsize=12)

                    
                ax.plot(amp_first_rank, amp_inferred, lw=2, ls='-', c=gm_color[gm], label='Mean ' + r'$\sigma$' ' = ' + str(round(gm, 3)))
                ax.set_xlabel('True amplitude of oscillating focal OTU', fontsize=11)

                if method_idx == 0:
                    ax.legend(loc='upper left', fontsize=8)

                #if method_idx == 3:
                #    ax.legend(loc='upper left', fontsize=8)


                if method_idx + rank_idx == 2:
                    ax.set_ylim([-0.05, 0.58])


    fig.subplots_adjust(hspace=0.3 , wspace=0.3)
    fig_name = "%soscillation_sim_results.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_oscillation_artifact_simulation_afd(method='clr'):

    param_dict = pickle.load(open(param_oscillation_artifact_simulation_path, "rb"))

    #ax_idx_all = [(0,0), (0,1), (0,2), (0,3), (0,4)]

    gm_all = list(param_dict[method]['nonfocal'].keys())
    sine_param_combo_all = list(param_dict[method]['nonfocal'][gm_all[0]].keys())

    fig = plt.figure(figsize = (4*len(sine_param_combo_all), 4*len(gm_all)))

    legend_elements = [Line2D([0], [0], color='dodgerblue', lw=2, label='One simulation')]

    for sine_param_combo_i_idx, sine_param_combo_i in enumerate(sine_param_combo_all):

        for gm_i_idx, gm_i in enumerate(gm_all):

            ax = plt.subplot2grid((len(gm_all), len(sine_param_combo_all)), (gm_i_idx, sine_param_combo_i_idx))

            afd_all = param_dict[method]['nonfocal'][gm_i][sine_param_combo_i]['afd']
            for afd in afd_all:
                ax.plot(days, afd, ls='-', lw=1, alpha=0.4, color='dodgerblue')


            ax.set_xlabel('Days', fontsize=10)
            ax.set_ylabel('%s abund. of non-oscillating taxon' % method_label_dict[method], fontsize=9)


            if gm_i_idx == 0:
                ax.set_title('Amp. of oscillating taxon = %.2f' % sine_param_combo_i[0], fontsize=11, fontweight='bold')

            if sine_param_combo_i_idx == 0:
                ax.text(-0.4, 0.5, "Std. dev of " + r'$\sigma^{2}$' + ' = %.2f' % (gm_i), rotation=90, fontsize=11, color='k', ha='center', va='center',  fontweight='bold', transform=ax.transAxes)

            
            if (gm_i_idx==0) and (sine_param_combo_i_idx == 0):
                ax.legend(handles=legend_elements, loc='upper left')


    fig.subplots_adjust(hspace=0.45, wspace=0.45)
    fig_name = "%stest_oscillation_sim_afd_nonfocal_%s.png" % (config.analysis_directory, method)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def mle_sigma(afd):

    log_mean_estimate = numpy.log(numpy.mean(afd))
    mean_log_estimate = numpy.mean(numpy.log(afd))

    def sigma_func(beta, mean_log_n, log_mean_n):
        return numpy.log(beta) - digamma(beta) - log_mean_n + mean_log_n
    
    s = log_mean_estimate - mean_log_estimate
    beta_init = (3 - s + numpy.sqrt((s-3)**2 + 24*s)) / (12*s)

    beta_estimate = fsolve(sigma_func, beta_init, args=(mean_log_estimate, log_mean_estimate))[0]
    sigma_estimate = 2/(beta_estimate + 1)

    if sigma_estimate >= 2:
        sigma_estimate = 1.99
        beta_estimate = (2-sigma_estimate)/sigma_estimate

    return beta_estimate, sigma_estimate




def make_compare_clr_to_true_abundance_dict(n_iter = 10):

    #This function  generates the parameters K and sigma for two communities with given parameters (mu, s),
    # S species, sigma^2 distributed as 'dist' (if dist is exponential, with average gm), correlation rho between the values of K
    # and correlation rhogamma between the Gamma-distributed fluctuations of abundance
    #sigma = 0.7
    #k = 0.01

    sigma_all = numpy.logspace(-2, numpy.log10(1), base=10, num=10)
    k_all = numpy.logspace(-4, -1, base=10, num=10)
    n_samples = 100
    n_reads = numpy.asarray([int(1e6)]*n_samples)

    mle_dict = {}

    for k in k_all:

        if k not in mle_dict:
            mle_dict[k] = {}

        for sigma in sigma_all:

            mle_dict[k][sigma] = {}
            mle_dict[k][sigma]['sigma_inferred_clr'] = []
            mle_dict[k][sigma]['sigma_inferred_rel'] = []
            #mle_dict[k][sigma]['k_inferred'] = []

            print(k, sigma)

            while len(mle_dict[k][sigma]['sigma_inferred_clr']) < n_iter:
            
                focal_x = gamma.rvs(numpy.divide(2,sigma)-1, scale=sigma*k/2, size=n_samples)

                # no relative abundances >= 1
                if sum(focal_x>=1) > 0:
                    continue

                nonfocal_x = 1 - focal_x

                rel_s_by_s = numpy.column_stack([focal_x, nonfocal_x]).T
                
                read_counts_multinomial_all = []
                for sad_idx, sad in enumerate(rel_s_by_s.T):
                    read_counts_multinomial_all.append(numpy.random.multinomial(n_reads[sad_idx], sad))

                read_counts_multinomial_all = numpy.asarray(read_counts_multinomial_all)

                # no absences
                if sum(read_counts_multinomial_all[:,0] == 0) > 0:
                    continue

                n_reads_gmean = gmean(read_counts_multinomial_all, axis=0)

                clr_s_by_s = numpy.log(read_counts_multinomial_all/n_reads_gmean)
                rel_s_by_s = (read_counts_multinomial_all.T/n_reads).T


                focal_clr = clr_s_by_s[0,:]
                focal_rel = rel_s_by_s[0,:]

                sigma_clr = mle_sigma(numpy.exp(focal_clr))
                sigma_rel = mle_sigma(focal_rel)

                mle_dict[k][sigma]['sigma_inferred_clr'].append(sigma_clr)
                mle_dict[k][sigma]['sigma_inferred_rel'].append(sigma_rel)
                #mle_dict[k][sigma]['k_inferred'].append(k_estiamte)

                print(k, sigma, numpy.absolute(sigma_clr - sigma)/sigma , numpy.absolute(sigma_rel - sigma)/sigma )

                # add inference from relative abundance......

    
    with open(compare_clr_to_true_abundance_dict_path, 'wb') as outfile:
        pickle.dump(mle_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)




def plot_compare_clr_to_true_abundance():

    mle_dict = pickle.load(open(compare_clr_to_true_abundance_dict_path, "rb"))
    
    k_all = list(mle_dict.keys())[:-1]
    k_all.sort()

    sigma_all = list(mle_dict[k_all[0]].keys())
    sigma_all.sort()

    sigma_all = numpy.asarray(sigma_all)

    rel_error_rel_all = []
    rel_error_clr_all = []

    for k in k_all:

        for sigma in sigma_all:

            sigma_inferred_rel = numpy.asarray(mle_dict[k][sigma]['sigma_inferred_rel'])
            sigma_inferred_clr = numpy.asarray(mle_dict[k][sigma]['sigma_inferred_clr'])

            rel_error_rel_all.append(numpy.absolute(numpy.mean(sigma_inferred_rel - sigma))/sigma)
            rel_error_clr_all.append(numpy.absolute(numpy.mean(sigma_inferred_clr - sigma))/sigma)

    
    rel_error_rel_all = numpy.asarray(rel_error_rel_all)
    rel_error_clr_all = numpy.asarray(rel_error_clr_all)

    array_merged = numpy.concatenate([rel_error_rel_all, rel_error_clr_all])
    min_, max_ = min(array_merged)*0.5, max(array_merged)*1.1

    
    fig = plt.figure(figsize = (4, 4))
    fig.subplots_adjust(bottom= 0.15)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

    ax.scatter(rel_error_rel_all, rel_error_clr_all, alpha=0.9, c='dodgerblue', s=20, zorder=2)

    ax.plot([min_, max_], [min_, max_], c='k', lw=2, ls=':', zorder=1)

    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)

    ax.set_xlabel("Mean relative error of " + r'$\sigma$' + ' inference, rel.', fontsize=10)
    ax.set_ylabel("Mean relative error of " + r'$\sigma$' + ' inference, CLR', fontsize=10)

    ax.set_xlim([min_,max_])
    ax.set_ylim([min_,max_])


    fig.subplots_adjust(hspace=0.35,wspace=0.4)
    fig_name = "%scompare_clr_to_true_abundance.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def make_compare_sigma_clr_to_true_abundance_oscillating_dict_old(n_iter=10):

    log_K_0 = -3
    amp = 1.5
    freq = 0.018
    phase = 1.8

    n_reads = numpy.asarray([int(1e6)]*len(days))

    # phase rangeing from yearily to daily oscillations..
    freq_range = numpy.linspace(2*numpy.pi/365, 2*numpy.pi, num=10)
    amp_range = numpy.linspace(0, 2, num=10)
    #sigma_all = numpy.logspace(-2, numpy.log10(1), base=10, num=10)
    sigma_all = numpy.linspace(0.1, 1.5, num=10)

    mle_dict = {}

    for freq in freq_range:

        mle_dict[freq] = {}

        for amp in amp_range:

            mle_dict[freq][amp] = {}

            for sigma in sigma_all:

                print(freq, amp, sigma)

                mle_dict[freq][amp][sigma] = {}
                mle_dict[freq][amp][sigma]['sigma_inferred_rel'] = []
                mle_dict[freq][amp][sigma]['sigma_inferred_clr'] = []

                K_t = numpy.exp((numpy.sin(numpy.outer(days, freq) + phase) * amp) + log_K_0)

                while len(mle_dict[freq][amp][sigma]['sigma_inferred_clr']) < n_iter:

                    focal_x = numpy.asarray([gamma.rvs(numpy.divide(2,sigma)-1, scale=sigma*K_t[idx]/2, size=1)[0] for idx in range(len(days))])

                    # no relative abundances >= 1
                    if sum(focal_x>=1) > 0:
                        continue

                    nonfocal_x = 1 - focal_x

                    rel_s_by_s = numpy.column_stack([focal_x, nonfocal_x]).T
                    
                    read_counts_multinomial_all = []
                    for sad_idx, sad in enumerate(rel_s_by_s.T):
                        read_counts_multinomial_all.append(numpy.random.multinomial(n_reads[sad_idx], sad))

                    read_counts_multinomial_all = numpy.asarray(read_counts_multinomial_all)

                    # no absences
                    if sum(read_counts_multinomial_all[:,0] == 0) > 0:
                        continue
                    
                    # length of vector is number of samples
                    n_reads_gmean = gmean(read_counts_multinomial_all, axis=1)

                    #clr_s_by_s = numpy.log(read_counts_multinomial_all/n_reads_gmean)
                    clr_s_by_s = (numpy.log(read_counts_multinomial_all).T - numpy.log(n_reads_gmean)).T
                    rel_s_by_s = (read_counts_multinomial_all.T/n_reads).T

                    focal_clr = clr_s_by_s[0,:]
                    focal_rel = rel_s_by_s[0,:]

                    sigma_clr = mle_sigma(numpy.exp(focal_clr))
                    sigma_rel = mle_sigma(focal_rel)

                    mle_dict[freq][amp][sigma]['sigma_inferred_clr'].append(sigma_clr)
                    mle_dict[freq][amp][sigma]['sigma_inferred_rel'].append(sigma_rel)


    with open(compare_sigma_clr_to_true_abundance_oscillating_dict_path, 'wb') as outfile:
        pickle.dump(mle_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)




def make_compare_sigma_clr_to_true_abundance_oscillating_dict(mu, s, S, N, n_sites, n_iter=1):

    #amp = 1.5
    #freq = 0.018
    focal_phase = 1.8

    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    focal_freq_range = numpy.linspace(0, 2*numpy.pi/90, num=5)
    focal_amp_range = numpy.linspace(0, 2, num=5)
    sigma_all = numpy.linspace(0.1, 1.5, num=10)

    #focal_amp_range = [focal_amp_range[-1]]

     # carrying capacity can be interpreted 
    # the logarithm of carrying capacities follow a sine wave
    # quenched for all 
    log_K_0 = numpy.random.normal(mu, s, S)
    # sort so that carrying capacities are increasing
    log_K_0 = numpy.sort(log_K_0)

    sys.stderr.write("Running simulations...\n")

    mle_dict = {}
    for focal_freq in focal_freq_range:

        mle_dict[focal_freq] = {}

        for focal_amp in focal_amp_range:

            mle_dict[focal_freq][focal_amp] = {}

            for sigma in sigma_all:

                print(focal_freq, focal_amp, sigma)

                mle_dict[focal_freq][focal_amp][sigma] = {}
                mle_dict[focal_freq][focal_amp][sigma]['focal_clr_afd'] = []
                mle_dict[focal_freq][focal_amp][sigma]['focal_inferred_sigma'] = []

                mle_dict[focal_freq][focal_amp][sigma]['nonfocal_clr_afd'] = []
                mle_dict[focal_freq][focal_amp][sigma]['nonfocal_inferred_sigma'] = []

                amp = numpy.repeat(0, repeats=S-1)
                amp = numpy.append(amp, focal_amp)

                freq = numpy.repeat(1, repeats=S-1)
                freq = numpy.append(freq, focal_freq)

                phase = numpy.repeat(0, repeats=S-1)
                phase = numpy.append(phase, focal_phase)

                # generate carrying capacity over time.
                K_t = numpy.exp((numpy.sin(numpy.outer(days, freq) + phase) * amp) + log_K_0)

                # same sigma for all OTUs
                #sigmarnd = numpy.repeat(sigma, S)
                # both focual and nonfocal have same sigma
                sigmarnd = numpy.repeat(0.5, repeats=S-2)
                sigmarnd = numpy.append(sigmarnd, sigma)
                sigmarnd = numpy.append(sigmarnd, sigma)

                while len(mle_dict[focal_freq][focal_amp][sigma]['focal_inferred_sigma']) < n_iter:
    
                    skip_iter = False

                    abundances_all, rel_abundances_all, read_counts_multinomial_all, read_counts_multinomial_all_nonzero, non_zero_idx = generate_community_from_sigma_k(S, K_t, sigmarnd, n_sites, N)
                    # relative abundance
                    rel_read_counts_multinomial_all_nonzero = read_counts_multinomial_all_nonzero/numpy.sum(read_counts_multinomial_all_nonzero, axis=0)
                    # all species are used to calculate relative abundance
                    clr_s_by_s, occupancy_idx = utils.clr_transform_sim_subset(read_counts_multinomial_all_nonzero, min_occupancy=1)
                    
                    S_sampled_clr = clr_s_by_s.shape[0]
                    if S_sampled_clr < 5:
                        skip_iter = True

                    if skip_iter == True:
                        continue
                

                    afd_iter_dict = {}

                    for otu_type in otu_type_all:

                        if otu_type == 'focal':
                            rank_idx = -1                        
                        else:
                            rank_idx = -2

                        #afd_otu = rel_abundances_all[rank_idx,:]
                        #afd_otu = abundances_all[rank_idx,:]
                        afd_clr_otu = clr_s_by_s[rank_idx,:]

                        # check for zeros in relative abundance
                        if sum(rel_read_counts_multinomial_all_nonzero[rank_idx,:] == 0) > 0:
                            skip_iter = True
                        
                        if sum(numpy.isnan(afd_clr_otu)) > 0:
                            skip_iter = True

                        afd_iter_dict[otu_type] = afd_clr_otu

                    # repeat process
                    if skip_iter == True:
                        continue

                    
                    for otu_type in otu_type_all:

                        afd_clr_otu = afd_iter_dict[otu_type]
                        
                        beta_estimate, sigma_estimate = mle_sigma(numpy.exp(afd_clr_otu))
                        mle_dict[focal_freq][focal_amp][sigma]['%s_clr_afd' % otu_type].append(afd_clr_otu.tolist())
                        mle_dict[focal_freq][focal_amp][sigma]['%s_inferred_sigma' % otu_type].append(sigma_estimate)
                        
                        #print(sigma_estimate)
                        #print(otu_type)
                        #print(mle_dict[focal_freq][focal_amp][sigma]['%s_inferred_sigma' % otu_type])

                        #if otu_type == 'focal':
                        #    print(sigma, sigma_estimate, len(mle_dict[focal_freq][focal_amp][sigma]['%s_inferred_sigma' % otu_type]))


    with open(compare_sigma_clr_to_true_abundance_oscillating_dict_path, 'wb') as outfile:
        pickle.dump(mle_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)





def plot_compare_clr_to_true_abundance_oscillating():

    mle_dict = pickle.load(open(compare_sigma_clr_to_true_abundance_oscillating_dict_path, "rb"))

    freq_all = list(mle_dict.keys())[:-1]
    freq_all.sort()

    amp_all = list(mle_dict[freq_all[0]].keys())
    amp_all.sort()

    simga_all = list(mle_dict[freq_all[0]][amp_all[0]].keys())
    simga_all.sort()

    fig = plt.figure(figsize = (12, 12))
    fig.subplots_adjust(bottom= 0.15)

    ax_all = []
    rel_error_clr_all = []

    for freq_idx, freq in enumerate(freq_all):

        for amp_idx, amp in enumerate(amp_all):

            #rel_error_rel_all = [numpy.mean(numpy.absolute(numpy.asarray(mle_dict[freq][amp][sigma]['sigma_inferred_rel']) - sigma)/sigma) for sigma in simga_all]
            focal_rel_error_clr_all = [numpy.mean(numpy.absolute(numpy.asarray(mle_dict[freq][amp][sigma]['focal_inferred_sigma']) - sigma)/sigma) for sigma in simga_all]
            nonfocal_rel_error_clr_all = [numpy.mean(numpy.absolute(numpy.asarray(mle_dict[freq][amp][sigma]['nonfocal_inferred_sigma']) - sigma)/sigma) for sigma in simga_all]

            rel_error_clr_all.extend(focal_rel_error_clr_all)
            rel_error_clr_all.extend(nonfocal_rel_error_clr_all)

            ax = plt.subplot2grid((len(freq_all), len(amp_all)), (freq_idx, amp_idx), colspan=1)
            ax_all.append(ax)

            ax.plot(simga_all, focal_rel_error_clr_all, c='#87CEEB', lw=2, label='Oscillating')
            ax.plot(simga_all, nonfocal_rel_error_clr_all, c='#FF6347', lw=2, label='Non-oscillating')

            #ax.set_title('Freq. = %.3f, Amp. = %.3f' % (freq, amp), fontsize=12)
            #ax.set_xscale('log', basex=10)
            
            ax.axhline(y=0, ls=':', lw=2, c='k')
            ax.xaxis.set_tick_params(labelsize=6)
            ax.yaxis.set_tick_params(labelsize=6)

            #ax.set_ylim([-0.05, ])

            if freq_idx + amp_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

            if freq_idx == 0:
                ax.set_title(r'$A_{i} = $' +  ' %.3f' % amp, fontsize=14)

            #if freq_idx == len(freq_all) - 1:
            #   # ax.set_xlabel("True " + r'$\sigma$', fontsize=14)                     

            if amp_idx == 0:
                #ax.set_ylabel("Mean rel. error of inferred " + r'$\sigma$', fontsize=6) 
                #ax.text(-0.5, 0.5, 'Freq. = %.3f' % freq, fontsize=10, ha='center', va='center', rotation=90, transform=ax.transAxes)
                
                if freq == 0:
                    tau = r'$\infty$'
                else:
                    tau = '%.0f' % (2*numpy.pi/freq)
                
                ax.set_ylabel(r'$\tau_{i}^{\mathrm{env}} = $' + ' ' + tau, fontsize=14)


            #if amp_idx == 0:
            #    y_label = 'Freq. = %.3f' % amp
            #    y_label = y_label + "\nMean rel. error of inferred " + r'$\sigma$'

            #else:
            #    y_label = "Mean rel. error of inferred " + r'$\sigma$'
            #ax.text(0.24, 0.8, r'$\rho^{2} = $' + str(round(rho**2, 3)), fontsize=15, ha='center', va='center', transform=ax.transAxes)
    
    
    fig.text(0.51, 0.1, "True " + r'$\sigma$', fontsize=20, ha='center', va='center')
    fig.text(0.05, 0.5, "Mean relative  error of inferred " + r'$\sigma$', fontsize=20, ha='center', va='center', rotation=90)

    
    for ax in ax_all:
        ax.set_xlim([0, max(simga_all)])
        ax.set_ylim([-0.05, max(rel_error_clr_all)])


    fig.subplots_adjust(hspace=0.15,wspace=0.2)
    fig_name = "%scompare_sigma_clr_to_true_abundance_oscillating.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.1, dpi = 600)
    plt.close()




def data_collapse_simulation(mu, s, S, N, dist, gm, n_sites, n_iter=1, rhogamma=0, n_otus_to_fit=30):

    param_dict = {}
    #param_dict['true_abundance'] = []
    param_dict['clr'] = []

    for p in sine_parameter_utils.param_no_method_all:
        param_dict['%s_mle' % p] = []


    #for gm in gm_all:
    #    param_dict['clr'][gm] = {}

    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    # carrying capacity can be interpreted 
    # the logarithm of carrying capacities follow a sine wave
    K = numpy.sort(numpy.exp(numpy.random.normal(mu, s, S)) )

    # sort so that carrying capacities are increasing
    sys.stderr.write("Running simulations...\n")
    #for n in range(n_iter):
    #while len(param_dict['clr'][gm]) < n_iter:

    #    skip_iter = False

        # we want all iterations to have the same sample of the sigma distribution
        # chack for all gm and all sine parameter combinations whether you get AFDs
        # where rank one and rank two have no zeros
    #    afd_iter_dict = {}
        # fix carrying capacity
    #    for gm in gm_all:

    #        afd_iter_dict[gm] = {}

    sigmarnd = []  #Exponentially distributed sigma, common for the two communities
    if dist == 'exp':
        for k in range(S):
            tr = 100
            while tr > 1.95: # Values too close to 2 give numerical problems when extracting from the Gamma distribution
                tr = numpy.sqrt(numpy.random.exponential(gm))
            sigmarnd.append(tr)

        sigmarnd = numpy.asarray(sigmarnd)
        # sigma defined on range 0 < sigma < 2

    if dist == 'unif':
        sigmarnd = numpy.random.uniform(0, 1.95, size=S)

    if dist == 'constant':
        sigmarnd = numpy.repeat(gm, S)


    abundances_all, rel_abundances_all, read_counts_multinomial_all, read_counts_multinomial_all_nonzero, non_zero_idx = generate_community_from_sigma_k(S, K, sigmarnd, n_sites, N)

    # relative abundance, all species present are used to calculate relative abundance
    rel_read_counts_multinomial_all_nonzero = read_counts_multinomial_all_nonzero/numpy.sum(read_counts_multinomial_all_nonzero, axis=0)

    
    clr_s_by_s, occupancy_idx = utils.clr_transform_sim(read_counts_multinomial_all_nonzero, min_occupancy=1)
    # get indiceces of must abundance species
    argsort_mean_rel_abund_idx = numpy.argsort(numpy.mean(rel_read_counts_multinomial_all_nonzero[occupancy_idx,:], axis=1))

    # sort clr_s_by_s by mean relative abundance
    clr_s_by_s = clr_s_by_s[argsort_mean_rel_abund_idx, :]
    clr_s_by_s_to_fit = clr_s_by_s[:n_otus_to_fit,:]
    
    # loop over OTUs
    for afd_clr_otu_idx in range(clr_s_by_s_to_fit.shape[0]):

        print(afd_clr_otu_idx)

        afd_clr_otu = clr_s_by_s_to_fit[afd_clr_otu_idx,:]
        exp_afd_clr_otu = numpy.exp(afd_clr_otu)

        freq_value = 2*numpy.pi/365 # 0.01721420632
        freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
        freq_max = 2*numpy.pi/180 # 0.034906585 (365-185)

        phase_value = numpy.pi
        phase_min = 0
        phase_max = 2*numpy.pi

        amp_value = 1
        amp_min = 1e-3
        amp_max = 10

        param_mean_value_exp_afd_clr = numpy.mean(exp_afd_clr_otu)
        param_min_value_exp_afd_clr = min(exp_afd_clr_otu)
        param_max_value_exp_afd_clr = max(exp_afd_clr_otu)

        params_afd_exp_clr = create_params(amp=dict(value=amp_value, min=amp_min, max=amp_max),
                                    freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                    phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                    param_mean=dict(value=param_mean_value_exp_afd_clr, min=param_min_value_exp_afd_clr, max=param_max_value_exp_afd_clr))


        beta_estimate_clr, sigma_estimate_clr = mle_sigma(exp_afd_clr_otu)
        result_brute_clr, fitter_clr = sine_parameter_utils.grid_search_mle_sine_wave(days, exp_afd_clr_otu, params_afd_exp_clr, beta_estimate_clr)


        best_result_mle_clr = sine_parameter_utils.second_round_optimization_mle(result_brute_clr, fitter_clr, beta_estimate_clr)
        best_params_mle_clr = best_result_mle_clr.params

        param_dict['clr'].append(afd_clr_otu.tolist())

        for p in sine_parameter_utils.param_no_method_all:
            param_dict['%s_mle' % p].append(best_params_mle_clr[p].value)

            print(best_params_mle_clr[p].value, param_dict['%s_mle' % p])

    

    sys.stderr.write("Saving parameter dictionary...\n")
    with open(data_collapse_simulation_path, 'wb') as outfile:
        pickle.dump(param_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")






def plot_oscillation_artifact_phase_simulation():

    fig = plt.figure(figsize = (8, 8))

    param_dict = pickle.load(open(param_oscillation_artifact_simulation_path, "rb"))

    for method_idx, method in enumerate(['log_rel', 'clr']):

        for rank_idx, rank in enumerate(['focal', 'nonfocal']):

            ax = plt.subplot2grid((2, 2), (method_idx, rank_idx))

            ax.text(-0.1, 1.07, utils.sub_plot_labels[method_idx+rank_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)

            phase_ticks = [0, 0.5*numpy.pi, numpy.pi, 1.5*numpy.pi, 2*numpy.pi]
            phase_tick_labels = [r'0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$',  r'$2\pi$']
            ax.set_yticks(phase_ticks)
            ax.set_yticklabels(phase_tick_labels)
            #ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=9)
            ax.set_ylim([0, 2*numpy.pi])

            ax.axhline(y=1.8, ls=':', lw=2, c='k', label='True phase of oscillating OTU')
            ax.axhline(y=1.8 + numpy.pi, ls='--', lw=2, c='k', label='True phase of oscillating OTU + ' + r'$\pi$' )


            for gm_idx, gm in enumerate(list(param_dict[method][rank].keys())):

                param_combo_all = list(param_dict[method][rank][gm].keys())

                amp_first_rank = [s[0] for s in param_combo_all]

                #sprint(param_combo_all)
                amp_inferred = [numpy.mean(param_dict[method][rank][gm][p]['phase_mle']) for p in param_combo_all]
                #print(amp_inferred)
                #print(amp_inferred)

                if gm_idx == 0:

                    if rank == 'focal':
                        #ax.plot([min(amp_first_rank), max(amp_first_rank)], [min(amp_first_rank), max(amp_first_rank)], ls=':', lw=2, c='k', label='1:1')
                        ax.set_ylabel('Inferred phase of oscillating OTU', fontsize=11)
                    else:
                        #ax.axhline(y=0, ls=':', lw=2, c='k', label='True amplitude of non-focal OTU')
                        ax.set_ylabel('Inferred phase of non-oscillating OTU', fontsize=11)


                    if method == 'log_rel':
                        ax.set_title('Relative abundance', fontsize=12)
                    else:
                        ax.set_title('CLR-transformed abundance', fontsize=12)

                    
                ax.plot(amp_first_rank, amp_inferred, lw=2, ls='-', c=gm_color[gm], label='Mean ' + r'$\sigma$' ' = ' + str(round(gm, 3)))
                ax.set_xlabel('True amplitude of oscillating focal OTU', fontsize=11)

                if (method_idx == 0) and (rank_idx==0):
                    ax.legend(loc='upper left', fontsize=8)

                #if method_idx + rank_idx == 2:
                #    ax.set_ylim([-0.05, 0.58])


    fig.subplots_adjust(hspace=0.3 , wspace=0.3)
    fig_name = "%soscillation_sim_phase_results.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()







if __name__ == "__main__":

    print("Running...")

    s = 3
    n_sites = len(days)
    S = 1000
    N = 100000
    mu = 0.001

    #plot_oscillation_artifact_simulation()

    #oscillation_artifact_simulation(mu, s, S, N, 'exp', [0.1, 0.3, 0.5], n_sites, focal_amp_all=[0, 0.5, 1, 1.5, 2], n_iter=10, clr_all_otus=False)

    #data_collapse_simulation(0.001, s, S, N, 'exp', 1, n_sites, n_iter=1)

    #test_amp_effect_fix_mean_var(0.0001, s, S, N, 'exp', 0.1, n_sites)


    #test_amp_effect_fix_mean_var_clr(0.0001, s, S, N, 'exp', 0.1, n_sites)


    #s_by_s, s_by_s_sampling = generate_community_oscillating_k(0.0001, s, S, N, 'exp', 1, n_sites, amp_focal=10)


    #make_compare_clr_to_true_abundance_dict()
    #plot_compare_clr_to_true_abundance()


    #make_compare_sigma_clr_to_true_abundance_oscillating_dict(mu, s, S, N, n_sites, n_iter=10)
    #plot_compare_clr_to_true_abundance_oscillating()
    
    #plot_oscillation_artifact_phase_simulation()

