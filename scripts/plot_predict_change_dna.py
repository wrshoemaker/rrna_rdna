import config
import sys
import pickle
import copy
import numpy
import utils
import scipy.stats as stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sine_parameter_utils

import matplotlib.pyplot as plt
from matplotlib import cm, colors

numpy.random.seed(123456789)


taxonomy_dict = utils.build_taxonomy_dict()



null_predict_change_dict_path = config.data_directory + 'null_predict_change_%sdict.pickle'

metadata_dict = utils.build_metadata_dict()
param_dict =  pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, 'rb'))



def get_null_predict_change_dict_path(otu_to_remove=None):

    if otu_to_remove == None:
        otu_to_remove_label = ''
    else:
        otu_to_remove_label = 'no_%s_' % otu_to_remove 

    null_predict_change_dict_path_ = null_predict_change_dict_path % (otu_to_remove_label)

    return null_predict_change_dict_path_




def make_null_predict_change_dict(n_perm = 10000, otu_to_remove=None):

    sys.stderr.write("Calculating observed correlations...\n")

    # *if* the RNA/DNA ratio is proportional to the growth rate, 
    # then we should be able to predict the change in DNA (proxy of biomass)....

    metadata_dict = utils.build_metadata_dict()
    s_by_s, otu_labels, samples = utils.load_count_data()
    clr_s_by_s_dna, clr_s_by_s_rna, occupancy_idx, otu_labels_subset, n_reads_dna_occupancy, n_reads_rna_occupancy = utils.clr_transform(s_by_s, otu_labels, samples)

    if otu_to_remove != None:
        otu_to_keep_idx = (otu_labels_subset != otu_to_remove)
        rel_s_by_s_dna = rel_s_by_s_dna[otu_to_keep_idx,:]
        rel_s_by_s_rna = rel_s_by_s_rna[otu_to_keep_idx,:]
        otu_labels_subset = otu_labels_subset[otu_to_keep_idx]


    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

    clr_s_by_s_rescaled_dna = (clr_s_by_s_dna.T - numpy.mean(clr_s_by_s_dna, axis=1)).T
    clr_s_by_s_rescaled_rna = (clr_s_by_s_rna.T - numpy.mean(clr_s_by_s_rna, axis=1)).T
    clr_s_by_s_rescaled_ratio = clr_s_by_s_rescaled_rna - clr_s_by_s_rescaled_dna

    diff_clr_s_by_s_rescaled_dna = clr_s_by_s_rescaled_dna[:,1:] - clr_s_by_s_rescaled_dna[:,:-1]
    diff_clr_s_by_s_rescaled_rna = clr_s_by_s_rescaled_rna[:,1:] - clr_s_by_s_rescaled_rna[:,:-1]
    time_idx_range = numpy.arange(len(days))

    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    samples_rna = samples[(sample_type=='RNA')]
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples_rna])
    delta_days = days[1:] - days[:-1]
    #env_variable_all = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph', 'air_temperature']
    env_variable_array = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples_rna])
    env_to_keep_idx = ~numpy.isnan(env_variable_array)

    null_predict_change_dict = {}
    for otu_i_idx, otu_i in enumerate(otu_labels_subset):
        
        null_predict_change_dict[otu_i] = {}
        null_predict_change_dict[otu_i]['slope_null_list'] = []  
        null_predict_change_dict[otu_i]['intercept_null_list'] = []  
        null_predict_change_dict[otu_i]['rho_null_list'] = []  

        clr_s_by_s_rescaled_ratio_i = clr_s_by_s_rescaled_ratio[otu_i_idx,:-1]
        diff_clr_s_by_s_rescaled_dna_i = diff_clr_s_by_s_rescaled_dna[otu_i_idx,:] / delta_days
        diff_clr_s_by_s_rescaled_rna_i = diff_clr_s_by_s_rescaled_rna[otu_i_idx,:] / delta_days

        null_predict_change_dict[otu_i]['clr_s_by_s_rescaled_ratio'] = clr_s_by_s_rescaled_ratio_i.tolist()
        null_predict_change_dict[otu_i]['diff_clr_s_by_s_rescaled_dna'] = diff_clr_s_by_s_rescaled_dna_i.tolist()
        null_predict_change_dict[otu_i]['diff_clr_s_by_s_rescaled_rna'] = diff_clr_s_by_s_rescaled_rna_i.tolist()
        null_predict_change_dict[otu_i]['n_reads_dna_occupancy_total'] = n_reads_dna_occupancy.tolist()
        null_predict_change_dict[otu_i]['n_reads_rna_occupancy_total'] = n_reads_dna_occupancy.tolist()

        # slope and intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(clr_s_by_s_rescaled_ratio_i, diff_clr_s_by_s_rescaled_dna_i)
        null_predict_change_dict[otu_i]['slope_obs'] = slope
        null_predict_change_dict[otu_i]['intercept_obs'] = intercept
        null_predict_change_dict[otu_i]['rho_obs'] = r_value

        # get environmental temperature correlation...
        rho_temp_clr_dna = numpy.corrcoef(clr_s_by_s_rescaled_dna[otu_i_idx,:][env_to_keep_idx], env_variable_array[env_to_keep_idx])[0,1]
        rho_temp_clr_rna = numpy.corrcoef(clr_s_by_s_rescaled_rna[otu_i_idx,:][env_to_keep_idx], env_variable_array[env_to_keep_idx])[0,1]

        null_predict_change_dict[otu_i]['rho_temp_clr_dna'] = rho_temp_clr_dna
        null_predict_change_dict[otu_i]['rho_temp_clr_rna'] = rho_temp_clr_rna
   


    sys.stderr.write("Generating distribution of null correlations via permuting time labels...\n")
    for n in range(n_perm):

        if n % 1000 == 0:

            sys.stderr.write("%d permutations done...\n" % n)

        time_idx_range_perm = numpy.random.permutation(time_idx_range)

        # permute both RNA and DNA timeseries
        clr_s_by_s_dna_null = clr_s_by_s_dna[:,time_idx_range_perm]
        clr_s_by_s_rna_null = clr_s_by_s_rna[:,time_idx_range_perm]

        # rescale null
        clr_s_by_s_dna_null_rescaled = (clr_s_by_s_dna_null.T - numpy.mean(clr_s_by_s_dna_null, axis=1)).T
        clr_s_by_s_rna_null_rescaled = (clr_s_by_s_rna_null.T - numpy.mean(clr_s_by_s_rna_null, axis=1)).T
        clr_s_by_s_ratio_null_rescaled = clr_s_by_s_rna_null_rescaled - clr_s_by_s_dna_null_rescaled

        diff_clr_s_by_s_dna_null_rescaled = clr_s_by_s_dna_null_rescaled[:,1:] - clr_s_by_s_dna_null_rescaled[:,:-1]

        for otu_i_idx, otu_i in enumerate(otu_labels_subset):

            clr_s_by_s_ratio_null_rescaled_i = clr_s_by_s_ratio_null_rescaled[otu_i_idx,:-1]
            diff_clr_s_by_s_dna_null_rescaled_i = diff_clr_s_by_s_dna_null_rescaled[otu_i_idx,:] / delta_days

            slope_null, intercept_null, r_value_null, p_value_null, std_err_null = stats.linregress(clr_s_by_s_ratio_null_rescaled_i, diff_clr_s_by_s_dna_null_rescaled_i)
            
            #null_predict_change_dict[otu_i]['rho_null_list'].append(rho_null)
            null_predict_change_dict[otu_i]['slope_null_list'].append(slope_null)
            null_predict_change_dict[otu_i]['intercept_null_list'].append(intercept_null)
            null_predict_change_dict[otu_i]['rho_null_list'].append(r_value_null)


    # calculate Z-scores
    for otu_i_idx, otu_i in enumerate(otu_labels_subset):
        for stat in ['slope', 'intercept', 'rho']:

            stat_null = numpy.asarray(null_predict_change_dict[otu_i]['%s_null_list' % stat])

            z_score_stat = (null_predict_change_dict[otu_i]['%s_obs' % stat] - numpy.mean(stat_null)) / numpy.std(stat_null)
            null_predict_change_dict[otu_i]['%s_z_score' % stat] = z_score_stat


    sys.stderr.write("Saving correlation dictionary...\n")

    null_predict_change_dict_path = get_null_predict_change_dict_path(otu_to_remove)    

    with open(null_predict_change_dict_path, 'wb') as outfile:
        pickle.dump(null_predict_change_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    sys.stderr.write("Done!\n")




def load_null_predict_change_dict_path_():
    dict_ = pickle.load(open(null_predict_change_dict_path, "rb"))
    return dict_


def load_null_predict_change_dict_path(otu_to_remove=None):
    dict_path = load_null_predict_change_dict_path_(otu_to_remove)
    dict_ = pickle.load(open(dict_path, "rb"))
    return dict_



def calculate_max_t(null_predict_change_dict, measure):

    otu_list = list(null_predict_change_dict.keys())

    null_otu_by_iter_matrix = numpy.asarray([null_predict_change_dict[i]['%s_null_list'% measure] for i in otu_list])

    max_value = numpy.max(null_otu_by_iter_matrix, axis=0)

    return max_value




def plot_predict_change_scatter(otu_to_remove=None):

    null_predict_change_dict = load_null_predict_change_dict_path(otu_to_remove)

    fig = plt.figure(figsize = (20, 20))
    fig.subplots_adjust(bottom= 0.15)

    #slope_null = calculate_max_t(null_predict_change_dict, 'rho')
    asv_count = 0

    otus_all = numpy.asarray(list(null_predict_change_dict.keys()))
    chunk_all = [otus_all[x:x+5] for x in range(0, len(otus_all), 5)]
    for chunk_idx, chunk in enumerate(chunk_all):

        for c_idx, c in enumerate(chunk):

            ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

            # [,:-1] because we're comparing it to the ratio of DNA abundances
            
            clr_s_by_s_rescaled_ratio_c = numpy.asarray(null_predict_change_dict[c]['clr_s_by_s_rescaled_ratio'])
            diff_clr_s_by_s_rescaled_dna_c = numpy.asarray(null_predict_change_dict[c]['diff_clr_s_by_s_rescaled_dna'])

            ax.scatter(clr_s_by_s_rescaled_ratio_c, diff_clr_s_by_s_rescaled_dna_c, s=8, alpha=1, c='k', zorder=2)
            ax.set_title(c, fontsize=11)

            ax.set_title('ASV %d (%s)' % (asv_count+1, taxonomy_dict[c]['family']), fontsize=11)
            ax.set_xlabel("RNA:DNA at time " + r'$t$' + ', ' + r'$\phi_{i}(t)$', fontsize=10)
            ax.set_ylabel("Per-day change in DNA, " + r'$\delta c_{i}^{\mathrm{DNA}} / \delta t $', fontsize=10)

            # regression
            #log10_mean_rescaled_ratio_over_otus_c = numpy.log10(mean_rescaled_ratio_over_otus_c)
            #log10_ratio_s_by_s_rescaled_dna_c = numpy.log10(ratio_s_by_s_rescaled_dna_c)
            slope, intercept, r_value, p_value, std_err = stats.linregress(clr_s_by_s_rescaled_ratio_c, diff_clr_s_by_s_rescaled_dna_c)
            
            x_range_ =  numpy.linspace(min(clr_s_by_s_rescaled_ratio_c), max(clr_s_by_s_rescaled_ratio_c), 10000)
            y_fit_range = slope*x_range_ + intercept

            null_slope_c = numpy.asarray(null_predict_change_dict[c]['slope_null_list'])
            #p_value_c = sum(null_slope_c > slope)/len(null_slope_c)
            print(numpy.mean(null_slope_c))

            p_value_c = utils.compute_pvalue(slope, null_slope_c, side="two")

            #print(c, p_value_c)

            if p_value_c <= 0.05:
                ax.plot(x_range_, y_fit_range, ls='--', lw=2.5, c='k')
                #print( r_value**2)

            ax.text(0.26, 0.87, r'$\beta_{1} = $' + str(round(slope, 3)), fontsize=12, ha='center', va='center', transform=ax.transAxes)
            #ax.text(0.26, 0.78, utils.get_p_value_latex_label_dict(p_value_c), fontsize=12, ha='center', va='center', transform=ax.transAxes)
            ax.text(0.26, 0.78, r'$P = $' + str(round(p_value_c, 4)), fontsize=12, ha='center', va='center', transform=ax.transAxes)

            asv_count += 1

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

    asv_count = 0

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
            ax.set_xlim([-1,1])

            ax.set_xlabel('Correlation coefficient', fontsize=10)
            ax.set_ylabel('Probability density', fontsize=10)
            ax.set_title('ASV %d (%s)' % (asv_count+1, taxonomy_dict[param_dict['otu_labels'][asv_count]]['family']), fontsize=11)


            if p_value <= 0.05:
                print(p_value)

            ax.text(0.24, 0.7, utils.get_p_value_latex_label_dict(p_value), fontsize=15, ha='center', va='center', transform=ax.transAxes)


            #ax.text(0.24, 0.8, r'$\rho^{2} = $' + str(round(rho**2, 3)), fontsize=15, ha='center', va='center', transform=ax.transAxes)

            if (c_idx==0) and (chunk_idx==0):
                ax.legend(loc='upper left')

            asv_count += 1


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





def plot_predict_change_vs_temp_rho():

    predict_dict = load_null_predict_change_dict_path()

    otu_to_plot = list(predict_dict.keys())

    rho_obs_all = [predict_dict[o]['rho_z_score'] for o in otu_to_plot]

    #rho_temp_clr_rna_all = [predict_dict[o]['rho_temp_clr_rna'] for o in otu_to_plot]
    #rho_temp_clr_dna_all = [predict_dict[o]['rho_temp_clr_dna'] for o in otu_to_plot]

    # autocorr correlation
    import plot_autocorrelation_otu
    autocorr_dict = pickle.load(open(plot_autocorrelation_otu.autocorrelation_dict_path, "rb"))


    autocorr_dna = [autocorr_dict['otu'][o]['DNA']['rho_autocorr_clr_vs_temp'] for o in otu_to_plot]
    autocorr_rna = [autocorr_dict['otu'][o]['RNA']['rho_autocorr_clr_vs_temp'] for o in otu_to_plot]

    fig = plt.figure(figsize = (8, 4))
    fig.subplots_adjust(bottom= 0.15)

    ax_dna = plt.subplot2grid((1, 2), (0, 0))
    ax_rna = plt.subplot2grid((1, 2), (0, 1))

    ax_dna.scatter(rho_obs_all, autocorr_dna, s=20, c=utils.dna_rna_color_dict['DNA'])
    ax_rna.scatter(rho_obs_all, autocorr_rna, s=20, c=utils.dna_rna_color_dict['RNA'])
    
    ax_dna.set_title('DNA', fontsize=14)
    ax_rna.set_title('RNA', fontsize=14)

    ax_dna.set_xlabel('Z-score of corr. b/w\nRNA/DNA(t) and DNA(t+delta t)', fontsize=10)
    ax_rna.set_xlabel('Z-score of corr. b/w\nRNA/DNA(t) and DNA(t+delta t)', fontsize=10)

    ax_dna.set_ylabel("Correlation between CLR and\nwater temp. autocorr. functions", fontsize=10)
    ax_rna.set_ylabel("Correlation between CLR and\nwater temp. autocorr. functions", fontsize=10)

    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig_name = "%spredict_change_vs_temp_rho.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")

    #make_null_predict_change_dict()

    plot_predict_change_scatter()  
    plot_predict_change_null_hist()


    #plot_predict_change_vs_temp_rho()


    #make_null_predict_change_dict(otu_to_remove='Otu000001')


