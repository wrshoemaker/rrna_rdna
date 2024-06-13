import config
import sys
import pickle
import copy
import numpy
import utils
import scipy.stats as stats


import matplotlib.pyplot as plt
from matplotlib import cm, colors

numpy.random.seed(123456789)




null_predict_change_dict_path = config.data_directory + 'null_predict_change_%sdict.pickle'


def get_null_predict_change_dict_path(otu_to_remove=None):

    if otu_to_remove == None:
        otu_to_remove_label = ''
    else:
        otu_to_remove_label = 'no_%s_' % otu_to_remove 

    null_predict_change_dict_path_ = null_predict_change_dict_path % (otu_to_remove_label)

    return null_predict_change_dict_path_



def make_null_predict_change_dict(n_perm = 10000, otu_to_remove=None):


    sys.stderr.write("Calculating observed correlations...\n")

    # discussion with Jacopo
    # *if* the RNA/DNA ratio is proportional to the growth rate, 
    # then we should be able to predict the change in DNA (proxy of biomass)....

    metadata_dict = utils.build_metadata_dict()
    s_by_s, otu_labels, samples = utils.load_count_data()
    rel_s_by_s_dna, rel_s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

    if otu_to_remove != None:
        otu_to_keep_idx = (otu_labels_subset != otu_to_remove)
        rel_s_by_s_dna = rel_s_by_s_dna[otu_to_keep_idx,:]
        rel_s_by_s_rna = rel_s_by_s_rna[otu_to_keep_idx,:]
        otu_labels_subset = otu_labels_subset[otu_to_keep_idx]


    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

    # returns rescaled relative abundance
    s_by_s_rescaled_dna = utils.rescale_s_by_s(rel_s_by_s_dna)
    s_by_s_rescaled_rna = utils.rescale_s_by_s(rel_s_by_s_rna)
    s_by_s_rescaled_ratio = s_by_s_rescaled_rna/s_by_s_rescaled_dna

    # average the ratio obser species
    mean_rescaled_ratio_over_otus = numpy.mean(s_by_s_rescaled_ratio, axis=0)
    s_by_s_rescaled_ratio_rescaled_by_otus = s_by_s_rescaled_ratio/mean_rescaled_ratio_over_otus
    s_by_s_rescaled_ratio_rescaled_by_otus_log10 = numpy.log10(s_by_s_rescaled_ratio_rescaled_by_otus)

    # change in relative abundance between timepoints
    ratio_s_by_s_rescaled_dna_log10 = numpy.log10(s_by_s_rescaled_dna[:,1:]/s_by_s_rescaled_dna[:,:-1])

    time_idx_range = numpy.arange(len(days))


    null_predict_change_dict = {}
    for otu_i_idx, otu_i in enumerate(otu_labels_subset):
        null_predict_change_dict[otu_i] = {}
        null_predict_change_dict[otu_i]['slope_null_list'] = []  
        null_predict_change_dict[otu_i]['intercept_null_list'] = []  
        null_predict_change_dict[otu_i]['rho_null_list'] = []  

        s_by_s_rescaled_ratio_rescaled_by_otus_log10_i = s_by_s_rescaled_ratio_rescaled_by_otus_log10[otu_i_idx,:-1]
        ratio_s_by_s_rescaled_dna_log10_i = ratio_s_by_s_rescaled_dna_log10[otu_i_idx,:]

        null_predict_change_dict[otu_i]['rescaled_rna_dna_ratio_log10'] = s_by_s_rescaled_ratio_rescaled_by_otus_log10_i.tolist()
        null_predict_change_dict[otu_i]['ratio_rescaked_dna_log10'] = ratio_s_by_s_rescaled_dna_log10_i.tolist()

        # slope and intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(s_by_s_rescaled_ratio_rescaled_by_otus_log10_i, ratio_s_by_s_rescaled_dna_log10_i)
        #null_predict_change_dict[otu_i]['rho_obs'] = numpy.corrcoef(s_by_s_rescaled_ratio_rescaled_by_otus_log10_i,ratio_s_by_s_rescaled_dna_log10_i )[0,1]
        null_predict_change_dict[otu_i]['slope_obs'] = slope
        null_predict_change_dict[otu_i]['intercept_obs'] = intercept
        null_predict_change_dict[otu_i]['rho_obs'] = r_value


    sys.stderr.write("Generating distribution of null correlations via permuting time labels...\n")
    for n in range(n_perm):

        if n % 1000 == 0:

            sys.stderr.write("%d permutations done...\n" % n)

        time_idx_range_perm = numpy.random.permutation(time_idx_range)

        s_by_s_rescaled_dna_null = s_by_s_rescaled_dna[:,time_idx_range_perm]
        ratio_s_by_s_rescaled_dna_log10_null = numpy.log10(s_by_s_rescaled_dna_null[:,1:]/s_by_s_rescaled_dna_null[:,:-1])

        # permute both RNA and DNA timeseries
        s_by_s_rescaled_rna_null = s_by_s_rescaled_rna[:,time_idx_range_perm]
        s_by_s_rescaled_ratio_null = s_by_s_rescaled_rna_null/s_by_s_rescaled_dna_null
        s_by_s_rescaled_ratio_rescaled_by_otus_null = s_by_s_rescaled_ratio_null/numpy.mean(s_by_s_rescaled_ratio_null, axis=0)
        s_by_s_rescaled_ratio_rescaled_by_otus_log10_null = numpy.log10(s_by_s_rescaled_ratio_rescaled_by_otus_null)

        for otu_i_idx, otu_i in enumerate(otu_labels_subset):

            mean_rescaled_ratio_over_otus_i_null = s_by_s_rescaled_ratio_rescaled_by_otus_log10_null[otu_i_idx,:-1]
            ratio_s_by_s_rescaled_dna_log10_null_i = ratio_s_by_s_rescaled_dna_log10_null[otu_i_idx,:]

            #rho_null = numpy.corrcoef(mean_rescaled_ratio_over_otus_i_null, ratio_s_by_s_rescaled_dna_log10_null_i)[0,1]
            slope_null, intercept_null, r_value_null, p_value_null, std_err_null = stats.linregress(mean_rescaled_ratio_over_otus_i_null, ratio_s_by_s_rescaled_dna_log10_null_i)
            
            #null_predict_change_dict[otu_i]['rho_null_list'].append(rho_null)
            null_predict_change_dict[otu_i]['slope_null_list'].append(slope_null)
            null_predict_change_dict[otu_i]['intercept_null_list'].append(intercept_null)
            null_predict_change_dict[otu_i]['rho_null_list'].append(r_value_null)


    #sys.stderr.write("%d permutations done...\n" % n)
    sys.stderr.write("Saving correlation dictionary...\n")

    null_predict_change_dict_path = get_null_predict_change_dict_path(otu_to_remove)    

    with open(null_predict_change_dict_path, 'wb') as outfile:
        pickle.dump(null_predict_change_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stderr.write("Done!\n")




def load_null_predict_change_dict_path():

    dict_ = pickle.load(open(null_predict_change_dict_path, "rb"))
    return dict_


def load_null_predict_change_dict_path(otu_to_remove=None):

    dict_path = get_null_predict_change_dict_path(otu_to_remove)

    dict_ = pickle.load(open(dict_path, "rb"))
    return dict_





def plot_predict_change_scatter(otu_to_remove=None):

    null_predict_change_dict = load_null_predict_change_dict_path(otu_to_remove)

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    #idx_all = list(range(len(otu_labels_subset)))
    otus_all = numpy.asarray(list(null_predict_change_dict.keys()))
    chunk_all = [otus_all[x:x+5] for x in range(0, len(otus_all), 5)]
    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            # [,:-1] because we're comparing it to the ratio of DNA abundances
            
            rescaled_rna_dna_ratio_log10 = numpy.asarray(null_predict_change_dict[c]['rescaled_rna_dna_ratio_log10'])
            ratio_rescaked_dna_log10 = numpy.asarray(null_predict_change_dict[c]['ratio_rescaked_dna_log10'])

            ax.scatter(10**rescaled_rna_dna_ratio_log10, 10**ratio_rescaked_dna_log10, s=8, alpha=1, c='k', zorder=2)
            ax.set_title(c, fontsize=11)
            ax.set_xlabel("Rescaled RNA:DNA ratio, " + r'$ \frac{\phi_{i}(t)}{\left< \phi(t) \right>_{S}}$', fontsize=10)
            ax.set_ylabel("Ratio of DNA relative\nabundances b/w timepoints, " + r'$ \frac{\tilde{x}_{i}^{d}(t + \delta t)}{\tilde{x}_{i}^{d}(t )}$', fontsize=10)

            # regression
            #log10_mean_rescaled_ratio_over_otus_c = numpy.log10(mean_rescaled_ratio_over_otus_c)
            #log10_ratio_s_by_s_rescaled_dna_c = numpy.log10(ratio_s_by_s_rescaled_dna_c)
            slope, intercept, r_value, p_value, std_err = stats.linregress(rescaled_rna_dna_ratio_log10, ratio_rescaked_dna_log10)
            
            x_range_ =  numpy.linspace(min(rescaled_rna_dna_ratio_log10), max(ratio_rescaked_dna_log10), 10000)
            y_fit_range = slope*x_range_ + intercept

            if p_value < 0.05:
                ax.plot(10**x_range_, 10**y_fit_range, ls='--', lw=2.5, c='k')
                print( r_value**2)

            ax.set_xscale('log', basex=10)
            ax.set_yscale('log', basey=10)

            #print(slope, r_value**2)

    if otu_to_remove == None:
        otu_to_remove_label = ''
    else:
        otu_to_remove_label = '_no_%s' % otu_to_remove 

    fig.subplots_adjust(hspace=0.35, wspace=0.45)
    fig_name = "%spredict_change_dna%s.png" % (config.analysis_directory, otu_to_remove_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_predict_change_null_hist():

    null_predict_change_dict = load_null_predict_change_dict_path()

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    #idx_all = list(range(len(otu_labels_subset)))
    otus_all = numpy.asarray(list(null_predict_change_dict.keys()))
    chunk_all = [otus_all[x:x+5] for x in range(0, len(otus_all), 5)]
    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            rho_null_array = numpy.asarray(null_predict_change_dict[c]['rho_null_list'])
            rho_obs = numpy.asarray(null_predict_change_dict[c]['rho_obs'])

            p_value = sum((rho_null_array>=rho_obs) / len(rho_null_array))
            
            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            #ax.hist(rho_null_array, bins=70, edgecolor='black', color='lightgray', density=True, zorder=1)
            ax.hist(rho_null_array, bins=50, color=utils.dna_rna_color_dict['DNA'], density=True, alpha=0.8, zorder=1, label='Null')
            ax.axvline(x=rho_obs, lw=3, ls='--', c='k', zorder=2, label='Observed')
            ax.set_xlim([-0.55,0.55])

            ax.set_xlabel('Correlation coefficient', fontsize=10)
            ax.set_ylabel('Probability density', fontsize=10)
            ax.set_title(c, fontsize=11)


            if p_value <= 0.05:
                print(p_value)

            ax.text(0.24, 0.7, utils.get_p_value_latex_label_dict(p_value), fontsize=15, ha='center', va='center', transform=ax.transAxes)


            #ax.text(0.24, 0.8, r'$\rho^{2} = $' + str(round(rho**2, 3)), fontsize=15, ha='center', va='center', transform=ax.transAxes)

            if (c_idx==0) and (chunk_idx==0):
                ax.legend(loc='upper left')

    fig.subplots_adjust(hspace=0.35, wspace=0.45)
    fig_name = "%spredict_change_null_hist.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_slope_comparison_no_otu1():

    predict_dict = load_null_predict_change_dict_path()
    predict_no_otu1_dict = load_null_predict_change_dict_path(otu_to_remove='Otu000001')

    otus_all = numpy.asarray(list(predict_dict.keys()))
    otus_no_otu1_all = numpy.asarray(list(predict_no_otu1_dict.keys()))
    otus_intersect = numpy.intersect1d(otus_all, otus_no_otu1_all)

    fig = plt.figure(figsize = (8, 4))
    fig.subplots_adjust(bottom= 0.15)

    label_list = ['slope', 'correlaton']

    for obs_idx, obs in enumerate(['slope_obs', 'rho_obs']):

        ax = plt.subplot2grid((1, 2), (0, obs_idx))

        obs_all = [predict_dict[i][obs] for i in otus_intersect]
        obs_no_otu1_all = [predict_no_otu1_dict[i][obs] for i in otus_intersect]

        merged_all = obs_all+obs_no_otu1_all
        min_ = min(merged_all)
        max_ = max(merged_all)

        ax.scatter(obs_all, obs_no_otu1_all, alpha=0.7, zorder=2)
        ax.plot([min_, max_], [min_, max_], ls=':', lw=2, c='k', zorder=1)
        ax.set_xlim([min_, max_])
        ax.set_ylim([min_, max_])

        ax.set_xlabel('RNA/DNA at t vs. DNA at t+delta t %s' % label_list[obs_idx], fontsize=10)
        ax.set_ylabel('RNA/DNA at t vs. DNA at t+delta t %s, no OTU1' % label_list[obs_idx], fontsize=10)

    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%sstat_comparison_no_otu1.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()







#plot_predict_change_scatter()

#plot_predict_change_null_hist()
#make_null_predict_change_dict()
#make_null_predict_change_dict(otu_to_remove='Otu000001')
#make_plot()

#plot_predict_change_scatter()
#plot_predict_change_scatter(otu_to_remove='Otu000001')



plot_slope_comparison_no_otu1()
