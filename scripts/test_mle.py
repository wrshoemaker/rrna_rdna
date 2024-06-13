import config
import numpy
import utils
import plot_sine_parameters
import mle_utils

import matplotlib.pyplot as plt


sample_to_analyze_all = ['DNA', 'RNA']

fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

#ax_richness = plt.subplot2grid((1, 2), (0, 0), colspan=1)
#ax_evenness = plt.subplot2grid((1, 2), (0, 1), colspan=1)

param_dict = plot_sine_parameters.load_param_dict(log10_status=True)
param_dict_no_otu1 = plot_sine_parameters.load_param_dict(log10_status=True, otu_to_remove='Otu000001')

s_by_s, otu_labels, samples = utils.load_count_data()

metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

# loop through RNA and DNA
for sample_to_analyze_idx, sample_to_analyze in enumerate(sample_to_analyze_all):

    #sample_to_analyze = 'DNA'
    # indices for sample to analyze
    s_by_s_sample_type_idx = (sample_type == sample_to_analyze)
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[s_by_s_sample_type_idx]])
    s_by_s_sample_to_analyse = s_by_s[:,s_by_s_sample_type_idx]
    n_reads = numpy.sum(s_by_s_sample_to_analyse, axis=0)

    # number of reads excluding dominant OTU
    otu1_idx = numpy.where(otu_labels=='Otu000001')[0][0]
    s_by_s_sample_to_analyze_no_otu1 = numpy.delete(s_by_s_sample_to_analyse, otu1_idx, axis=0)
    n_reads_no_otu1 = numpy.sum(s_by_s_sample_to_analyze_no_otu1, axis=0)

    n_reads_all = [n_reads, n_reads_no_otu1]
    param_dict_all = [param_dict, param_dict_no_otu1]

    rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

    if sample_to_analyze == 'DNA':
        mean_rel_sample_to_analyze = numpy.mean(rel_s_by_s_dna, axis=1)
    elif sample_to_analyze == 'RNA':
        mean_rel_sample_to_analyze = numpy.mean(rel_s_by_s_rna, axis=1)

    delta_l_nested = [[],[]]
    
    for otu_i_idx, otu_i in enumerate(otu_labels_subset):

        if otu_i == 'Otu000001':
            continue
        
        # same read distribution
        otu_i_s_by_s_idx = numpy.where(otu_labels==otu_i)[0][0]
        afd_reads_i = s_by_s_sample_to_analyse[otu_i_s_by_s_idx,:]

        # loop through OTU1
        for j_idx in range(2):

            param_dict_j = param_dict_all[j_idx]
            n_reads_j = n_reads_all[j_idx]

            otu_i_dict_idx = param_dict_j['otu']['otu_labels'].index(otu_i)
            amp = param_dict_j['otu']['amp_leastsq'][sample_to_analyze][otu_i_dict_idx]
            freq = param_dict_j['otu']['freq_leastsq'][sample_to_analyze][otu_i_dict_idx]
            phase = param_dict_j['otu']['phase_leastsq'][sample_to_analyze][otu_i_dict_idx]
            mean_log = param_dict_j['otu']['param_mean_leastsq'][sample_to_analyze][otu_i_dict_idx]
            
            # model was fit to the log of the rescaled data. 
            # need to rescale
            mean = mean_rel_sample_to_analyze[otu_i_idx]
            k = numpy.exp(mean_log) * mean

            init_params = (numpy.mean(afd_reads_i/n_reads_j), (numpy.mean(afd_reads_i/n_reads_j)/numpy.std(afd_reads_i/n_reads_j))**2)

            # MLE for negative binomial, no
            gamma_sampling_model = mle_utils.mle_gamma_param_sampling(n_reads_j, afd_reads_i)
            gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", args=init_params, disp = False, bounds= [(0.0000001,1), (0.0001,1.9999)])
            gamma_sampling_model_ll = gamma_sampling_model.loglike(gamma_sampling_result.params)

            #print(gamma_sampling_result.params[0])
            #print(gamma_sampling_result.hessv)


            afd_reads_and_t = numpy.transpose((afd_reads_i, days))

            # MLE for negative binomial, fixing parameters from sine wave
            # k, sigma, amp, freq, phase
            gamma_param_sine_sampling_model = mle_utils.mle_gamma_param_sine_sampling(n_reads_j, afd_reads_and_t)
            gamma_param_sine_sampling_result = gamma_param_sine_sampling_model.fit(method="lbfgs", disp = False, bounds= [(k*0.001, 1), (0.000001, 1.9999), (amp, amp), (freq, freq), (phase, phase)])
            gamma_param_sine_sampling_ll = gamma_param_sine_sampling_model.loglike(gamma_param_sine_sampling_result.params)

            #print(gamma_param_sine_sampling_result.hessv)

            #print(gamma_sampling_result.params[0], gamma_param_sine_sampling_result.params[0])

            delta_l = 2*(gamma_param_sine_sampling_ll - gamma_sampling_model_ll)

            delta_l_nested[j_idx].append(delta_l)

            #print(k, delta_l)


    ax = plt.subplot2grid((1, 2), (0, sample_to_analyze_idx), colspan=1)

    min_max = [min(delta_l_nested[0] + delta_l_nested[1]), max(delta_l_nested[0] + delta_l_nested[1])]

    print(min_max)

    #print(delta_l_nested[0])
    #print(delta_l_nested[1])

    #print(min(delta_l_nested), max(min_max))

    #print(0, max(min_max))


    ax.scatter(delta_l_nested[0], delta_l_nested[1], s=15, alpha=0.9, c=utils.dna_rna_color_dict[sample_to_analyze], zorder=2)
    ax.plot([0,max(min_max)], [0,max(min_max)], ls=':', lw=1.5, c='k', zorder=1, label='1:1')
    ax.set_xlim(0, max(min_max))
    ax.set_ylim(0, max(min_max))
    ax.set_xlabel("Log-likelihood ratio", fontsize=10)
    ax.set_ylabel("Log-likelihood ratio, excluding OTU1", fontsize=10) 
    ax.set_title(sample_to_analyze)


 
fig.subplots_adjust(hspace=0.35, wspace=0.40)
fig_name = "%slikelihood_ratio_comparison.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()









