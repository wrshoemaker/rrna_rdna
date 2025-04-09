import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm


s_by_s, otu_labels, samples = utils.load_count_data()

metadata_dict = utils.build_metadata_dict()

sample_type_all = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

sample_type_set = ['RNA', 'DNA']



fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)


for sample_type_i_idx, sample_type_i in enumerate(sample_type_set):

    sample_type_all_i_idx = (sample_type_all==sample_type_i)

    sample_type_all_i = sample_type_all[sample_type_all_i_idx]

    s_by_s_i = s_by_s[:,sample_type_all_i_idx]

    occupancies, predicted_occupancies, mad, beta, species = utils.predict_occupancy(s_by_s_i, s_by_s_i.shape[0])

    idx_to_keep = (occupancies>0) & (predicted_occupancies > 0) & (mad > 0)
    mad = mad[idx_to_keep]
    occupancies = occupancies[idx_to_keep]
    predicted_occupancies = predicted_occupancies[idx_to_keep]

    mad_occupancy_joint = numpy.concatenate((mad, occupancies),axis=0)
    min_ = min(mad_occupancy_joint)
    max_ = max(mad_occupancy_joint)

    sorted_plot_data = utils.plot_color_by_pt_dens(mad, occupancies, radius=utils.color_radius, loglog=1)
    x,y,z = sorted_plot_data[:, 0], sorted_plot_data[:, 1], sorted_plot_data[:, 2]


    ax = plt.subplot2grid((1, 2), (0, sample_type_i_idx), colspan=1)
    ax.scatter(x, y, c=numpy.sqrt(z), cmap='Blues', s=70, alpha=0.9, edgecolors='none', zorder=1)
    # edgecolors='none',
    all_ = numpy.concatenate([x, y])


    # mad vs occupancy
    mad_log10 = numpy.log10(mad)
    occupancies_log10 = numpy.log10(occupancies)
    predicted_occupancies_log10 = numpy.log10(predicted_occupancies)
    hist_all, bin_edges_all = numpy.histogram(mad_log10, density=True, bins=25)
    bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_mean_all_to_keep = []
    bins_occupancies = []
    for i in range(0, len(bin_edges_all)-1 ):
        predicted_occupancies_log10_i = predicted_occupancies_log10[(mad_log10>=bin_edges_all[i]) & (mad_log10<bin_edges_all[i+1])]
        #bins_mean_all_to_keep.append(bins_mean_all[i])
        bins_mean_all_to_keep.append(bin_edges_all[i])
        bins_occupancies.append(numpy.mean(predicted_occupancies_log10_i))


    bins_mean_all_to_keep = numpy.asarray(bins_mean_all_to_keep)
    bins_occupancies = numpy.asarray(bins_occupancies)

    bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_occupancies))]
    bins_occupancies_no_nan = bins_occupancies[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_occupancies))]

    ax.plot(10**bins_mean_all_to_keep_no_nan, 10**bins_occupancies_no_nan, lw=3, ls='--',c='k', zorder=2, label='Gamma prediction')

    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)
    #ax.set_xlabel('Mean relative abundance', fontsize=11)
    #ax.set_ylabel('Occupancy', fontsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    ax.tick_params(axis='both', which='major', labelsize=9)

    ax.set_xlabel("Mean relative abundance", fontsize = 10)
    ax.set_ylabel("Occupancy", fontsize = 10)

    ax.legend(loc="lower right", fontsize=8)


    ax.set_title(sample_type_i, fontsize=14)





fig.subplots_adjust(hspace=0.35,wspace=0.25)
fig_name = "%sadundance_vs_occupancy.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()