import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from scipy import stats


taxonomic_level = 'genus'
taxonomy_dict = utils.build_taxonomy_dict()
rrna_copy_dict = utils.make_rrna_copy_dict()
rrna_copy_taxa = numpy.asarray(list(rrna_copy_dict.keys()))

sample_types_all = ['RNA', 'DNA']


s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)
# s_by_s.shape = (246, 134265)

metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])


fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

for i_idx, i in enumerate(sample_types_all):

    sample_type_i_idx = (sample_type==i)
    sample_type_i = samples[sample_type_i_idx]
    rel_s_by_s_i = rel_s_by_s[:,sample_type_i_idx]

    #samples_dna_copy_number_i, days_copy_number_i, copy_number_coarse_labels_i, mean_copy_number_i = utils.calculate_mean_copy_number(i, taxonomic_level)

    #otu_to_keep_i = []
    #for key_i, value_i in taxonomy_dict.items():

    #    if value_i['genus'] in copy_number_coarse_labels_i:
    #        otu_to_keep_i.append(key_i)

    #otu_to_keep_i = numpy.unique(otu_to_keep_i)
    #otu_to_keep_i_idx = numpy.asarray([numpy.where(otu_labels==j)[0][0] for j in otu_to_keep_i])

    #rel_s_by_s_i = rel_s_by_s_i[otu_to_keep_i_idx,:]
    #otu_labels_i = otu_labels[otu_to_keep_i_idx]

    # then coarse-grain.
    rel_s_by_s_i_coarse, coarse_labels_i, n_coarse, taxa_to_keep_idx = utils.coarse_grain_abundances_by_taxonomy(rel_s_by_s_i, otu_labels, taxonomic_level)    
    mad_i_coarse = numpy.mean(rel_s_by_s_i_coarse, axis=1)

    coarse_labels_to_keep = numpy.intersect1d(rrna_copy_taxa, coarse_labels_i)
    coarse_labels_to_keep = numpy.sort(coarse_labels_to_keep)

    mad_to_plot_idx = numpy.asarray([numpy.where(coarse_labels_i==k)[0][0] for k in coarse_labels_to_keep])
    mad_to_plot = mad_i_coarse[mad_to_plot_idx]
    copy_number_to_plot = numpy.asarray([rrna_copy_dict[c] for c in coarse_labels_to_keep])


    non_zero_idx = (mad_to_plot>0)
    mad_to_plot = mad_to_plot[non_zero_idx]
    copy_number_to_plot = copy_number_to_plot[non_zero_idx]

    slope, intercept, r_value, p_value, std_err = stats.linregress(copy_number_to_plot, numpy.log10(mad_to_plot))
    x_range_ =  numpy.linspace(min(copy_number_to_plot) , max(copy_number_to_plot) , 10000)
    y_fit_range = slope*x_range_ + intercept

    print(slope, r_value**2, p_value)

    #ax.scatter(mean_trajectory_ratio_i_subset, mean_copy_number_i, s=5, alpha=0.8, c=utils.dna_rna_color_dict[type_i], zorder=1, label=type_i)


    ax = plt.subplot2grid((1, 2), (0, i_idx), colspan=1)
    ax.scatter(copy_number_to_plot, mad_to_plot, s=7, alpha=0.5)
    ax.plot(x_range_, 10**y_fit_range, lw=1.5, ls='--', zorder=1, c='k')

    ax.set_xlim([min(copy_number_to_plot), max(copy_number_to_plot)])
    ax.set_ylim([min(mad_to_plot), max(mad_to_plot)])

    ax.set_yscale('log', basey=10)
    ax.set_xlabel("rRNA copy number", fontsize=10)
    ax.set_ylabel('Mean relative abundance, %s' % i, fontsize=10)




fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%smad_vs_copy_number.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

