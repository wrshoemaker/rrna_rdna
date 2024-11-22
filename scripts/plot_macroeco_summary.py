import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats, signal
# numdifftools also installed
import pickle


min_occupany_afd = 0.8


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])
delta_days = days[1:] - days[:-1]




fig = plt.figure(figsize = (8, 12.5))
fig.subplots_adjust(bottom= 0.15)


for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

    sample_type_idx = (sample_type==data_type)
    s_by_s_data = s_by_s[:,sample_type_idx]
    n_reads = numpy.sum(s_by_s_data, axis=0)
    rel_s_by_s_data = s_by_s_data/n_reads

    rel_s_by_s_data_afd_idx = (numpy.sum(rel_s_by_s_data>0, axis=1)/len(n_reads)) >= min_occupany_afd
    rel_s_by_s_data_afd = rel_s_by_s_data[rel_s_by_s_data_afd_idx,:]


    ax_afd = plt.subplot2grid((3, 2), (0, data_type_idx))
    ax_mad = plt.subplot2grid((3, 2), (1, data_type_idx))
    ax_taylors = plt.subplot2grid((3, 2), (2, data_type_idx))

    ax_afd.text(-0.1, 1.07, utils.sub_plot_labels[data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_afd.transAxes)
    ax_mad.text(-0.1, 1.07, utils.sub_plot_labels[2+data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_mad.transAxes)
    ax_taylors.text(-0.1, 1.07, utils.sub_plot_labels[4+data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax_taylors.transAxes)


    ax_afd.set_title(data_type, fontsize=12, color=utils.dna_rna_color_dict[data_type], fontweight='bold')



    # plot AFD
    rescaled_afd_log10_all = []
    hist_to_plot_all = []
    for afd in rel_s_by_s_data_afd:

        afd = afd[afd>0]
        afd_log10 = numpy.log10(afd)
        rescaled_afd_log10 = (afd_log10 - numpy.mean(afd_log10))/numpy.std(afd_log10)

        hist_to_plot, bins_mean_to_plot = utils.get_hist_and_bins(rescaled_afd_log10, bins=12)
        ax_afd.scatter(bins_mean_to_plot, hist_to_plot, s=5, color=utils.dna_rna_color_dict[data_type], alpha=0.3, lw=1)
        #ax_afd.plot(bins_mean_to_plot, hist_to_plot, lw=0.5, color=utils.dna_rna_color_dict[data_type], alpha=0.3, zorder=1)

        rescaled_afd_log10_all.append(rescaled_afd_log10)
        hist_to_plot_all.append(hist_to_plot)


    rescaled_afd_log10_all = numpy.concatenate(rescaled_afd_log10_all)
    hist_to_plot_all = numpy.concatenate(hist_to_plot_all)
    
    # fit loggamma
    shape_gamma, loc_gamma, scale_gamma = stats.loggamma.fit(rescaled_afd_log10_all)
    x = numpy.linspace(stats.loggamma.ppf(0.001, shape_gamma, loc=loc_gamma, scale=scale_gamma), stats.loggamma.ppf(0.999, shape_gamma, loc=loc_gamma, scale=scale_gamma), 100)
    pdf_loggamma_to_plot = stats.loggamma.pdf(x, shape_gamma, loc=loc_gamma, scale=scale_gamma)
    ax_afd.plot(x, pdf_loggamma_to_plot, 'k', ls='--', lw=3, label='Gamma fit')

    ax_afd.set_ylim([min(hist_to_plot_all), max(hist_to_plot_all)])
    ax_afd.set_yscale('log', basey=10)
    ax_afd.set_xlabel("Rescaled " + r'$\mathrm{log}_{10}$'  + " relative abundance", fontsize = 11)
    ax_afd.set_ylabel("Probability density", fontsize = 11)

    # plot mean vs occupancy
    occupancies, predicted_occupancies, mad, beta, species = utils.predict_occupancy(s_by_s_data, otu_labels)

    idx_to_keep = (occupancies>0) & (predicted_occupancies > 0) & (mad > 0)
    mad = mad[idx_to_keep]
    occupancies = occupancies[idx_to_keep]
    predicted_occupancies = predicted_occupancies[idx_to_keep]

    mad_occupancy_joint = numpy.concatenate((mad, occupancies),axis=0)
    min_ = min(mad_occupancy_joint)
    max_ = max(mad_occupancy_joint)

    sorted_plot_data = utils.plot_color_by_pt_dens(mad, occupancies, radius=utils.color_radius, loglog=1)
    x,y,z = sorted_plot_data[:, 0], sorted_plot_data[:, 1], sorted_plot_data[:, 2]

    ax_mad.scatter(x, y, c=numpy.sqrt(z), cmap=utils.cmap_data_type_dict[data_type], s=70, alpha=0.9, edgecolors='none', zorder=1)
    #all_ = numpy.concatenate([x, y])
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

    ax_mad.plot(10**bins_mean_all_to_keep_no_nan, 10**bins_occupancies_no_nan, lw=3, ls='--',c='k', zorder=2, label='Gamma prediction')

    ax_mad.set_xscale('log', basex=10)
    ax_mad.set_yscale('log', basey=10)
    #ax_mad.tick_params(axis='both', which='minor', labelsize=9)
    #ax_mad.tick_params(axis='both', which='major', labelsize=9)

    ax_mad.set_xlabel("Mean relative abundance", fontsize = 10)
    ax_mad.set_ylabel("Occupancy", fontsize = 10)


    # plot taylors law
    rel_s_by_s_data_taylor_idx = (numpy.sum(rel_s_by_s_data>0, axis=1)/len(n_reads)) > 0 
    rel_s_by_s_data_taylor = rel_s_by_s_data[rel_s_by_s_data_taylor_idx,:]
    mean_all = numpy.mean(rel_s_by_s_data_taylor, axis=1)
    var_all = numpy.var(rel_s_by_s_data_taylor, axis=1)

    taylors_slope, taylors_intercept, taylors_r_value, taylors_p_value, taylors_std_err = stats.linregress(numpy.log10(mean_all), numpy.log10(var_all))
    
    x_taylors_range = numpy.linspace(min(numpy.log10(mean_all)), max(numpy.log10(mean_all)), num=1000)
    #y_predict_taylors = taylors_intercept + 2*x_taylors_range
    y_predict = taylors_intercept + taylors_slope*x_taylors_range


    ax_taylors.scatter(mean_all, var_all, s=3, color=utils.dna_rna_color_dict[data_type], alpha=0.3, lw=1, zorder=1)
    #ax_taylors.plot(10**x_taylors_range, 10**y_predict_taylors, lw=2, ls='-', c='k')
    ax_taylors.plot(10**x_taylors_range, 10**y_predict, lw=2, ls='--', c='k', zorder=2)


    ax_taylors.set_xlim([min(mean_all), max(mean_all)])
    ax_taylors.set_ylim([min(var_all), max(var_all)])

    ax_taylors.set_xscale('log', basex=10)
    ax_taylors.set_yscale('log', basey=10)

    ax_taylors.set_xlabel("Mean relative abundance", fontsize = 11)
    ax_taylors.set_ylabel("Variance of relative abundance", fontsize = 11)

    text_label = r'$\sigma_{{{}}}^{{{}}} \sim  \bar{{{}}}^{{{}}}$'.format('x', '2', 'x', str( round(taylors_slope, 3) ))

    ax_taylors.text(0.2,0.9, text_label, fontsize=12, color='k', ha='center', va='center', transform=ax_taylors.transAxes  )
    ax_taylors.tick_params(axis='both', which='minor', labelsize=9)
    #ax_taylors.tick_params(axis='both', which='major', labelsize=9)

    # plot logfold
    log_ratio = numpy.log10(rel_s_by_s_data_afd[:,1:]/rel_s_by_s_data_afd[:,:-1])

    for log_ratio_i in log_ratio:

        log_ratio_to_plot_idx = (numpy.isfinite(log_ratio_i)) & (~numpy.isnan(log_ratio_i))
        #log_ratio_i = log_ratio_i[log_ratio_to_plot_idx]
        log_ratio_per_day_i = log_ratio_i[log_ratio_to_plot_idx]/delta_days[log_ratio_to_plot_idx]

        hist_log_ratio_i, bins_log_ratio_i = utils.get_hist_and_bins(log_ratio_per_day_i, bins=12)
        #ax_logfold.scatter(bins_log_ratio_i, hist_log_ratio_i, s=7, color=utils.dna_rna_color_dict[data_type], alpha=0.3, lw=1)

    
    #ax_logfold.axvline(x=0, ls=':', c='k', lw=2, zorder=2, label='Stationarity')
    #ax_logfold.set_yscale('log', basey=10, nonposy='clip')
    #ax_logfold.set_xlabel("Log-fold abundance ratio, " + r'$\Delta \ell_{x}$', fontsize=10)
    #ax_logfold.set_ylabel("Probability density", fontsize=10)


    if data_type_idx == 0:
        ax_afd.legend(loc="upper left", fontsize=8)
        ax_mad.legend(loc="lower right", fontsize=8)
        #ax_logfold.legend(loc="upper left", fontsize=8)






fig.subplots_adjust(hspace=0.3, wspace=0.35)
fig_name = "%smacroeco_summary.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



