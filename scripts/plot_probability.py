import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)

metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

sample_type_rna_idx = (sample_type=='RNA')
sample_type_dna_idx = (sample_type=='DNA')

sample_type_rna = samples[sample_type_rna_idx]

rel_s_by_s_rna = rel_s_by_s[:,sample_type_rna_idx]
rel_s_by_s_dna = rel_s_by_s[:,sample_type_dna_idx]

occupancy_rna = numpy.sum(rel_s_by_s_rna>0, axis=1)/rel_s_by_s_rna.shape[1]
occupancy_dna = numpy.sum(rel_s_by_s_dna>0, axis=1)/rel_s_by_s_dna.shape[1]




def plot_probability_conditioned_dna():

    #hist_to_plot, bins_mean_to_plot = utils.get_hist_and_bins(occupancy_dna)
    hist, bin_edges = numpy.histogram(occupancy_dna, density=True, bins=50)
    hist = hist/sum(hist)

    bin_mean = numpy.asarray([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )])

    prob_rna_present_all = []
    for lower_bin_idx in range(len(bin_edges)-1):

        lower_bin = bin_edges[lower_bin_idx]
        upper_bin = bin_edges[lower_bin_idx+1]

        # include zeros
        if lower_bin_idx == 0:
            occupancy_dna_bin_idx = (occupancy_dna>=lower_bin) & (occupancy_dna<=upper_bin)

        else:
            occupancy_dna_bin_idx = (occupancy_dna>lower_bin) & (occupancy_dna<=upper_bin)

        # subset RNA occupancies within the DNA occupancy bin
        occupancy_rna_bin = occupancy_rna[occupancy_dna_bin_idx]
        
        # fraction of detected OTUs within the bin
        prob_rna_present = sum(occupancy_rna_bin>0)/len(occupancy_rna_bin)
        prob_rna_present_all.append(prob_rna_present)


    prob_rna_present_all = numpy.asarray(prob_rna_present_all)


    # repeat for log of DNA occupancy
    # remove zeros for log transforming
    occupancy_rna_detected_dna = occupancy_rna[(occupancy_dna>0)]
    occupancy_dna_detected_dna = occupancy_dna[(occupancy_dna>0)]

    log_occupancy_dna_detected_dna = numpy.log10(occupancy_dna_detected_dna)


    hist_log, bin_edges_log = numpy.histogram(log_occupancy_dna_detected_dna, density=True, bins=50)
    hist_log = hist_log/sum(hist_log)
    bin_mean_log = numpy.asarray([0.5 * (bin_edges_log[i] + bin_edges_log[i+1]) for i in range(0, len(bin_edges_log)-1 )])


    bin_mean_log_to_plot = []
    prob_rna_present_log_all = []
    prob_rna_present_log_all_with_zero = []
    for lower_bin_idx in range(len(bin_edges_log)-1):

        lower_bin = bin_edges_log[lower_bin_idx]
        upper_bin = bin_edges_log[lower_bin_idx+1]

        if lower_bin_idx == 0:
            log_occupancy_dna_bin_idx = (log_occupancy_dna_detected_dna>=lower_bin) & (log_occupancy_dna_detected_dna<=upper_bin)

        else:
            log_occupancy_dna_bin_idx = (log_occupancy_dna_detected_dna>lower_bin) & (log_occupancy_dna_detected_dna<=upper_bin)


        occupancy_rna_bin = occupancy_rna_detected_dna[log_occupancy_dna_bin_idx]
        
        # fraction of detected OTUs within the bin
        if len(occupancy_rna_bin) < 10:
            prob_rna_present_log_all_with_zero.append(0)
            continue

        else:

            prob_rna_present_log = sum(occupancy_rna_bin>0)/len(occupancy_rna_bin)
            prob_rna_present_log_all.append(prob_rna_present_log)

            bin_mean_log_to_plot.append(bin_mean_log[lower_bin_idx])

            prob_rna_present_log_all_with_zero.append(prob_rna_present_log)



    bin_mean_log_to_plot = numpy.asarray(bin_mean_log_to_plot)
    prob_rna_present_log_all = numpy.asarray(prob_rna_present_log_all)
    prob_rna_present_log_all_with_zero = numpy.asarray(prob_rna_present_log_all_with_zero)


    fig = plt.figure(figsize = (8, 8))
    fig.subplots_adjust(bottom= 0.15)

    ax_density = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax_prob = plt.subplot2grid((2, 2), (0, 1), colspan=1)

    ax_density_log = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax_prob_log = plt.subplot2grid((2, 2), (1, 1), colspan=1)



    ax_density.scatter(bin_mean[hist>0], hist[hist>0], s=7, color='k', alpha=0.9, lw=1, label=r'$P(o_{\mathrm{DNA}})$')
    ax_density.scatter(bin_mean[prob_rna_present_all>0], prob_rna_present_all[prob_rna_present_all>0], s=7, color='dodgerblue', alpha=0.9, lw=1, label=r'$P( o_{\mathrm{RNA}} > 0 | o_{\mathrm{DNA}})$')

    ax_density.set_xlabel("Occupancy, DNA", fontsize = 10)
    ax_density.set_ylabel("Probability Density", fontsize = 10)
    ax_density.legend(loc="center right")
    ax_density.set_yscale('log', basey=10)


    #ax_prob
    prob_to_plot_idx = (prob_rna_present_all>0) & (hist>0)
    ax_prob.scatter(hist[prob_to_plot_idx], prob_rna_present_all[prob_to_plot_idx], s=7, color='k', alpha=0.9, lw=1, label=r'$P(o_{\mathrm{DNA}})$')

    ax_prob.set_xscale('log', basex=10)
    #ax_prob.set_yscale('log', basey=10)
    ax_prob.set_xlabel(r'$P(o_{\mathrm{DNA}})$', fontsize = 12)
    ax_prob.set_ylabel(r'$P( o_{\mathrm{RNA}} > 0 | o_{\mathrm{DNA}})$', fontsize = 12)


    # ax_density_log
    ax_density_log.scatter(bin_mean_log[hist_log>0], hist_log[hist_log>0], s=7, color='k', alpha=0.9, lw=1, label=r'$P(o_{\mathrm{DNA}})$')
    ax_density_log.scatter(bin_mean_log_to_plot, prob_rna_present_log_all, s=7, color='dodgerblue', alpha=0.9, lw=1, label=r'$P( o_{\mathrm{RNA}} > 0 | o_{\mathrm{DNA}})$')

    ax_density_log.set_xlabel("Log of occupancy, DNA", fontsize = 10)
    ax_density_log.set_ylabel("Probability Density", fontsize = 10)
    ax_density.legend(loc="center right")
    ax_density_log.set_yscale('log', basey=10)


    # ax_prob_log
    prob_log_to_plot_idx = (prob_rna_present_log_all_with_zero>0) & (hist_log>0)
    ax_prob_log.scatter(hist_log[prob_log_to_plot_idx], prob_rna_present_log_all_with_zero[prob_log_to_plot_idx], s=7, color='k', alpha=0.9, lw=1, label=r'$P(o_{\mathrm{DNA}})$')

    ax_prob_log.set_xscale('log', basex=10)
    #ax_prob.set_yscale('log', basey=10)
    ax_prob_log.set_xlabel(r'$P(o_{\mathrm{DNA}})$', fontsize = 12)
    ax_prob_log.set_ylabel(r'$P( o_{\mathrm{RNA}} > 0 | o_{\mathrm{DNA}})$', fontsize = 12)


    fig.subplots_adjust(hspace=0.35,wspace=0.25)
    fig_name = "%sprobability_conditioned_dna.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()







def plot_probability_conditioned(conditioned_on='rna'):

    if conditioned_on == 'rna':

        conditioned_variable = occupancy_rna
        response_variable = occupancy_dna
        prob_label = r'$P(o_{\mathrm{RNA}})$'
        prob_conditioned_label = r'$P( o_{\mathrm{DNA}} > 0 | o_{\mathrm{RNA}})$'

    else:

        conditioned_variable = occupancy_dna
        response_variable = occupancy_rna
        prob_label = r'$P(o_{\mathrm{DNA}})$'
        prob_conditioned_label = r'$P( o_{\mathrm{RNA}} > 0 | o_{\mathrm{DNA}})$'


    hist, bin_edges = numpy.histogram(conditioned_variable, density=True, bins=50)
    hist = hist/sum(hist)

    bin_mean = numpy.asarray([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(0, len(bin_edges)-1 )])

    prob_response_nonzero_all = []
    for lower_bin_idx in range(len(bin_edges)-1):

        lower_bin = bin_edges[lower_bin_idx]
        upper_bin = bin_edges[lower_bin_idx+1]

        # include zeros
        if lower_bin_idx == 0:
            conditioned_variable_bin_idx = (conditioned_variable>=lower_bin) & (conditioned_variable<=upper_bin)

        else:
            conditioned_variable_bin_idx = (conditioned_variable>lower_bin) & (conditioned_variable<=upper_bin)

        # subset RNA occupancies within the DNA occupancy bin
        response_variable_bin = response_variable[conditioned_variable_bin_idx]
        
        # fraction of detected OTUs within the bin
        prob_response_nonzero = sum(response_variable_bin>0)/len(response_variable_bin)
        prob_response_nonzero_all.append(prob_response_nonzero)


    prob_response_nonzero_all = numpy.asarray(prob_response_nonzero_all)


    # repeat for log of DNA occupancy
    # remove zeros for log transforming
    response_variable_detected_conditioned = response_variable[(conditioned_variable>0)]
    conditioned_variable_detected_conditioned = conditioned_variable[(conditioned_variable>0)]

    log_conditioned_variable_detected_conditioned = numpy.log10(conditioned_variable_detected_conditioned)


    hist_log, bin_edges_log = numpy.histogram(log_conditioned_variable_detected_conditioned, density=True, bins=50)
    hist_log = hist_log/sum(hist_log)
    bin_mean_log = numpy.asarray([0.5 * (bin_edges_log[i] + bin_edges_log[i+1]) for i in range(0, len(bin_edges_log)-1 )])

    bin_mean_log_to_plot = []
    prob_response_nonzero_log_all = []
    prob_response_nonzero_log_all_with_zero = []
    for lower_bin_idx in range(len(bin_edges_log)-1):

        lower_bin = bin_edges_log[lower_bin_idx]
        upper_bin = bin_edges_log[lower_bin_idx+1]

        if lower_bin_idx == 0:
            log_conditioned_variable_bin_idx = (log_conditioned_variable_detected_conditioned>=lower_bin) & (log_conditioned_variable_detected_conditioned<=upper_bin)

        else:
            log_conditioned_variable_bin_idx = (log_conditioned_variable_detected_conditioned>lower_bin) & (log_conditioned_variable_detected_conditioned<=upper_bin)


        response_variable_bin = response_variable_detected_conditioned[log_conditioned_variable_bin_idx]
        
        # fraction of detected OTUs within the bin
        if len(response_variable_bin) < 10:
            prob_response_nonzero_log_all_with_zero.append(0)
            continue

        else:

            prob_response_nonzero_log = sum(response_variable_bin>0)/len(response_variable_bin)
            prob_response_nonzero_log_all.append(prob_response_nonzero_log)

            bin_mean_log_to_plot.append(bin_mean_log[lower_bin_idx])

            prob_response_nonzero_log_all_with_zero.append(prob_response_nonzero_log)



    bin_mean_log_to_plot = numpy.asarray(bin_mean_log_to_plot)
    prob_response_nonzero_log_all = numpy.asarray(prob_response_nonzero_log_all)
    prob_response_nonzero_log_all_with_zero = numpy.asarray(prob_response_nonzero_log_all_with_zero)


    fig = plt.figure(figsize = (8, 8))
    fig.subplots_adjust(bottom= 0.15)

    ax_density = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax_prob = plt.subplot2grid((2, 2), (0, 1), colspan=1)

    ax_density_log = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax_prob_log = plt.subplot2grid((2, 2), (1, 1), colspan=1)



    ax_density.scatter(bin_mean[hist>0], hist[hist>0], s=7, color='k', alpha=0.9, lw=1, label=prob_label)
    ax_density.scatter(bin_mean[prob_response_nonzero_all>0], prob_response_nonzero_all[prob_response_nonzero_all>0], s=7, color='dodgerblue', alpha=0.9, lw=1, label=prob_conditioned_label)

    ax_density.set_xlabel("Occupancy, %s" % conditioned_on.upper(), fontsize = 10)
    ax_density.set_ylabel("Probability Density", fontsize = 10)
    ax_density.legend(loc="center right")
    ax_density.set_yscale('log', basey=10)


    #ax_prob
    prob_to_plot_idx = (prob_response_nonzero_all>0) & (hist>0)
    ax_prob.scatter(hist[prob_to_plot_idx], prob_response_nonzero_all[prob_to_plot_idx], s=7, color='k', alpha=0.9, lw=1, label=prob_label)

    ax_prob.set_xscale('log', basex=10)
    #ax_prob.set_yscale('log', basey=10)
    ax_prob.set_xlabel(prob_label, fontsize = 12)
    ax_prob.set_ylabel(prob_conditioned_label, fontsize = 12)


    # ax_density_log
    ax_density_log.scatter(bin_mean_log[hist_log>0], hist_log[hist_log>0], s=7, color='k', alpha=0.9, lw=1, label=prob_label)
    ax_density_log.scatter(bin_mean_log_to_plot, prob_response_nonzero_log_all, s=7, color='dodgerblue', alpha=0.9, lw=1, label=prob_conditioned_label)

    ax_density_log.set_xlabel("Log of occupancy, %s" % conditioned_on.upper(), fontsize = 10)
    ax_density_log.set_ylabel("Probability Density", fontsize = 10)
    ax_density.legend(loc="center right")
    ax_density_log.set_yscale('log', basey=10)


    # ax_prob_log
    prob_log_to_plot_idx = (prob_response_nonzero_log_all_with_zero>0) & (hist_log>0)
    ax_prob_log.scatter(hist_log[prob_log_to_plot_idx], prob_response_nonzero_log_all_with_zero[prob_log_to_plot_idx], s=7, color='k', alpha=0.9, lw=1, label=r'$P(o_{\mathrm{DNA}})$')

    ax_prob_log.set_xscale('log', basex=10)
    #ax_prob.set_yscale('log', basey=10)
    ax_prob_log.set_xlabel(prob_label, fontsize = 12)
    ax_prob_log.set_ylabel(prob_conditioned_label, fontsize = 12)


    fig.subplots_adjust(hspace=0.35,wspace=0.25)
    fig_name = "%sprobability_conditioned_%s.png" % (config.analysis_directory, conditioned_on)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




#plot_probability_conditioned_dna()

plot_probability_conditioned('rna')
plot_probability_conditioned('dna')


#print(bin_edges_)

#print(len(hist_), len(bin_edges_))

