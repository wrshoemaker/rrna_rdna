import os
import sys
import pickle
import numpy
from scipy.special import polygamma
from scipy.stats import norm
from scipy.stats import gamma
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


method_label_dict = {'log_rel': 'Rescaled log rel.', 'clr': 'CLR'}


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

#s = 3
s = 3
n_sites = len(days)
S = 1000
N = 100000


def generate_community_from_sigma_k(k, sigma, n_sites, N, rhogamma=0):

    # Extraction of the who vectors of abundances, distributed according to Gamma distributions with the correlation rhogamma
    cov = numpy.ones((n_sites, n_sites))
    I = numpy.identity(n_sites)
    cov = ((cov-I)*rhogamma) + I

    Z = numpy.random.multivariate_normal(numpy.asarray([0]*n_sites), cov, S, tol=1e-5)
    U = norm.cdf(Z)

    abundances_all = [gamma.ppf(U[:,idx], numpy.divide(2,sigma)-1, scale=sigma*k[idx,:]/2) for idx in range(n_sites)]
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

    return rel_abundances_all, read_counts_multinomial_all, read_counts_multinomial_all_nonzero, non_zero_idx




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





def oscillation_artifact_simulation(mu, s, S, N, dist, gm_all, n_sites, focal_amp_all=[4], focal_freq_all=[2*numpy.pi/365], focal_phase_all=[1.8], n_iter=1, rhogamma=0):

    # Fix distribution of K_0
    # Loop through iteration
    # For each iteration draw sigmas, and abundanaces from distribution using each sine parameter combination
    # Fit model
    # get parameters

    otu_type_all = ['focal', 'nonfocal']

    sine_param_combo_all = list(itertools.product(focal_amp_all, focal_freq_all, focal_phase_all))

    param_dict = {}
    param_dict['clr'] = {}
    param_dict['log_rel'] = {}

    for otu_type in otu_type_all:
        param_dict['clr'][otu_type] = {}
        param_dict['log_rel'][otu_type] = {}
    
        for gm in gm_all:
            param_dict['clr'][otu_type][gm] = {}
            param_dict['log_rel'][otu_type][gm] = {}

            for sine_param_combo in sine_param_combo_all:
                param_dict['clr'][otu_type][gm][sine_param_combo] = {}
                param_dict['log_rel'][otu_type][gm][sine_param_combo] = {}

                param_dict['clr'][otu_type][gm][sine_param_combo]['num_sampled_species'] = []
                param_dict['log_rel'][otu_type][gm][sine_param_combo]['num_sampled_species'] = []

                param_dict['clr'][otu_type][gm][sine_param_combo]['afd'] = []
                param_dict['log_rel'][otu_type][gm][sine_param_combo]['afd'] = []

                for p in sine_parameter_utils.param_no_method_all:
                    param_dict['clr'][otu_type][gm][sine_param_combo]['%s_leastsq' % p] = []
                    param_dict['log_rel'][otu_type][gm][sine_param_combo]['%s_leastsq' % p] = []


    if type(N) == int:
        N = numpy.asarray([N]*n_sites)

    # carrying capacity can be interpreted 
    # the logarithm of carrying capacities follow a sine wave
    log_K_0 = numpy.random.normal(mu, s, S)
    # sort so that carrying capacities are increasing
    log_K_0 = numpy.sort(log_K_0)

    sys.stderr.write("Running simulations...\n")
    #for n in range(n_iter):
    while len(param_dict['clr'][otu_type][gm][sine_param_combo]['amp_leastsq']) < n_iter:

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

                rel_abundances_all, read_counts_multinomial_all, read_counts_multinomial_all_nonzero, non_zero_idx = generate_community_from_sigma_k(K_t, sigmarnd, n_sites, N)

                # relative abundance
                rescaled_rel_s_by_s = utils.rescale_s_by_s(read_counts_multinomial_all_nonzero)
                # all species are used to calculate relative abundance

                # CLR
                clr_s_by_s, occupancy_idx = utils.clr_transform_sim(read_counts_multinomial_all_nonzero, min_occupancy=1)
                rescaled_clr_s_by_s = (clr_s_by_s.T - numpy.mean(clr_s_by_s, axis=1)).T

                afd_iter_dict[gm][sine_param_combo] = {}
                afd_iter_dict[gm][sine_param_combo]['num_sampled_species_log_rel'] = read_counts_multinomial_all_nonzero.shape[0]
                afd_iter_dict[gm][sine_param_combo]['num_sampled_species_clr'] = sum(occupancy_idx)

                for otu_type in otu_type_all:

                    afd_iter_dict[gm][sine_param_combo][otu_type] = {}

                    if otu_type == 'focal':
                        rank_idx = -1                        
                    else:
                        rank_idx = -2

                    afd_clr_rank_2 = rescaled_clr_s_by_s[rank_idx,:]
                    afd_log_rel_rank_2 = numpy.log10(rescaled_rel_s_by_s[rank_idx,:])

                    # check for zeros in relative abundance
                    if sum(rescaled_rel_s_by_s[rank_idx,:] == 0) > 0:
                        skip_iter = True
                    
                    if sum(numpy.isnan(afd_clr_rank_2)) > 0:
                        skip_iter = True

                    afd_iter_dict[gm][sine_param_combo][otu_type]['clr'] = afd_clr_rank_2
                    afd_iter_dict[gm][sine_param_combo][otu_type]['log_rel'] = afd_log_rel_rank_2

        # repeat process
        if skip_iter == True:
            continue
        
        # proceed...
        for gm in gm_all:

            for sine_param_combo in sine_param_combo_all:

                focal_amp, focal_freq, focal_phase = sine_param_combo[0], sine_param_combo[1], sine_param_combo[2]
                sys.stderr.write("Parameter sigma exp = %.2f, Amp = %.2f, Freq = %.4f, Phase = %.3f, %s, Iter = %d ...\n" % (gm, focal_amp, focal_freq, focal_phase, otu_type, len(param_dict['clr'][otu_type][gm][sine_param_combo]['amp_leastsq'])))

                param_dict['log_rel'][otu_type][gm][sine_param_combo]['num_sampled_species'].append(afd_iter_dict[gm][sine_param_combo]['num_sampled_species_log_rel'])
                param_dict['clr'][otu_type][gm][sine_param_combo]['num_sampled_species'].append(afd_iter_dict[gm][sine_param_combo]['num_sampled_species_clr'])

                for otu_type in otu_type_all:
                    
                    afd_clr_rank_2 = afd_iter_dict[gm][sine_param_combo][otu_type]['clr']
                    afd_log_rel_rank_2 = afd_iter_dict[gm][sine_param_combo][otu_type]['log_rel']

                    freq_value = 2*numpy.pi/365 # 0.01721420632
                    freq_min = 2*numpy.pi/550 # 0.01142397328 (365+185)
                    freq_max = 2*numpy.pi/180 # 0.034906585 (365-185)

                    phase_value = numpy.pi
                    phase_min = 0
                    phase_max = 2*numpy.pi

                    amp_value_clr = 1
                    amp_min_clr = 1e-3
                    amp_max_clr = 40

                    param_mean_min_clr = -2
                    param_mean_max_clr = 2
                    param_mean_value_clr = numpy.mean(afd_clr_rank_2)
                    param_mean_value_log_rel = numpy.mean(afd_log_rel_rank_2)

                    amp_value_log_rel = 1
                    amp_min_log_rel = 1e-3
                    amp_max_log_rel = 3

                    param_mean_min_log_rel = -0.5
                    param_mean_max_log_rel = 0.5


                    params_clr = create_params(amp=dict(value=amp_value_clr, min=amp_min_clr, max=amp_max_clr),
                                                freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                                phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                                param_mean=dict(value=param_mean_value_clr, min=param_mean_min_clr, max=param_mean_max_clr))

                    params_log_rel = create_params(amp=dict(value=amp_value_log_rel, min=amp_min_log_rel, max=amp_max_log_rel),
                                                freq=dict(value=freq_value, min=freq_min, max=freq_max),
                                                phase=dict(value=phase_value, min=phase_min, max=phase_max),
                                                param_mean=dict(value=param_mean_value_log_rel, min=param_mean_min_log_rel, max=param_mean_max_log_rel))


                    result_brute_clr, fitter_clr = sine_parameter_utils.grid_search_sine_wave(days, afd_clr_rank_2, params_clr)
                    result_brute_log_rel, fitter_log_rel = sine_parameter_utils.grid_search_sine_wave(days, afd_log_rel_rank_2, params_log_rel)

                    best_result_leastsq_clr = sine_parameter_utils.second_rount_optimization(result_brute_clr, fitter_clr)
                    best_result_leastsq_log_rel = sine_parameter_utils.second_rount_optimization(result_brute_log_rel, fitter_log_rel)

                    best_params_leastsq_clr = best_result_leastsq_clr.params
                    best_params_leastsq_log_rel = best_result_leastsq_log_rel.params

                    param_dict['clr'][otu_type][gm][sine_param_combo]['afd'].append(afd_clr_rank_2.tolist())
                    param_dict['log_rel'][otu_type][gm][sine_param_combo]['afd'].append(afd_log_rel_rank_2.tolist())

                    for p in sine_parameter_utils.param_no_method_all:
                        param_dict['clr'][otu_type][gm][sine_param_combo]['%s_leastsq' % p].append(best_params_leastsq_clr[p].value)
                        param_dict['log_rel'][otu_type][gm][sine_param_combo]['%s_leastsq' % p].append(best_params_leastsq_log_rel[p].value)

                        if otu_type == 'nonfocal':

                            if p == 'amp':
                                print(best_params_leastsq_clr[p].value, best_params_leastsq_log_rel[p].value)



    sys.stderr.write("Saving parameter dictionary...\n")
    with open(param_oscillation_artifact_simulation_path, 'wb') as outfile:
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

    gm_color = {0.1:'lightskyblue', 0.3:'dodgerblue', 0.5:'royalblue'}

    fig = plt.figure(figsize = (8, 8))

    param_dict = pickle.load(open(param_oscillation_artifact_simulation_path, "rb"))

    for method_idx, method in enumerate(['log_rel', 'clr']):

        for rank_idx, rank in enumerate(['focal', 'nonfocal']):

            ax = plt.subplot2grid((2, 2), (method_idx, rank_idx))

            for gm_idx, gm in enumerate(list(param_dict[method][rank].keys())):

                param_combo_all = list(param_dict[method][rank][gm].keys())

                amp_first_rank = [s[0] for s in param_combo_all]

                amp_inferred = [numpy.mean(param_dict[method][rank][gm][p]['amp_leastsq']) for p in param_combo_all]


                if gm_idx == 0:

                    if rank == 'focal':
                        ax.plot([min(amp_first_rank), max(amp_first_rank)], [min(amp_first_rank), max(amp_first_rank)], ls=':', lw=2, c='k', label='1:1')
                        ax.set_ylabel('Inferred amplitude of focal OTU', fontsize=11)
                    else:
                        ax.axhline(y=0, ls=':', lw=2, c='k', label='True amplitude of non-focal OTU')
                        ax.set_ylabel('Inferred amplitude of non-focal OTU', fontsize=11)


                    if method == 'log_rel':
                        ax.set_title('Log relative abundance', fontsize=12)
                    else:
                        ax.set_title('CLR abundance', fontsize=12)

                    
                ax.plot(amp_first_rank, amp_inferred, lw=2, ls='-', c=gm_color[gm], label='Std. dev of ' + r'$\sigma^{2}$' ' dist. = ' + str(round(gm, 3)))
                ax.set_xlabel('True amplitude of oscillating focal OTU', fontsize=11)


                if method_idx == 0:
                    ax.legend(loc='upper left', fontsize=8)



    fig.subplots_adjust(hspace=0.3 , wspace=0.3)
    fig_name = "%stest_oscillation_sim.png" % config.analysis_directory
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

        








if __name__ == "__main__":

    print("Running...")

    #oscillation_artifact_simulation(0.001, s, S, N, 'exp', [0.1, 0.3, 0.5], n_sites, focal_amp_all=[0, 0.5, 1, 1.5, 2], n_iter=10)

    plot_oscillation_artifact_simulation()

    #plot_oscillation_artifact_simulation_afd(method='clr')
    #plot_oscillation_artifact_simulation_afd(method='log_rel')

    #test_amp_effect_fix_mean_var(0.0001, s, S, N, 'exp', 0.1, n_sites)


    #test_amp_effect_fix_mean_var_clr(0.0001, s, S, N, 'exp', 0.1, n_sites)


    #s_by_s, s_by_s_sampling = generate_community_oscillating_k(0.0001, s, S, N, 'exp', 1, n_sites, amp_focal=10)

