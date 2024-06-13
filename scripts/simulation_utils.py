import os
import numpy
from scipy.special import polygamma
from scipy.stats import norm
from scipy.stats import gamma
import utils
import config

import matplotlib.pyplot as plt
from matplotlib import cm

numpy.random.seed(123456789)

s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

#s = 3
s = 3
n_sites = len(days)
S = 1000
N = 100000


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
    #read_counts_poisson_all = numpy.asarray(read_counts_poisson_all).T


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
    print(amp_focal_range)

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



test_amp_effect_fix_mean_var(0.0001, s, S, N, 'exp', 0.1, n_sites)


#s_by_s, s_by_s_sampling = generate_community_oscillating_k(0.0001, s, S, N, 'exp', 1, n_sites, amp_focal=10)

