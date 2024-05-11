import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats, signal


from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')



numpy.random.seed(123456789)


s_by_s, otu_labels, samples = utils.load_count_data()
rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)
# s_by_s.shape = (246, 134265)

metadata_dict = utils.build_metadata_dict()

sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

sample_type_rna_idx = (sample_type=='RNA')
sample_type_dna_idx = (sample_type=='DNA')

sample_type_rna = samples[sample_type_rna_idx]

rel_s_by_s_rna = rel_s_by_s[:,sample_type_rna_idx]
rel_s_by_s_dna = rel_s_by_s[:,sample_type_dna_idx]


#days = numpy.asarray([metadata_dict[s]['day'] for s in sample_type_rna])
#water_temp = numpy.asarray([metadata_dict[s]['water_temp'] for s in sample_type_rna])
#water_temp_non_nan_idx = ~numpy.isnan(water_temp)
#water_temp_non_nan = water_temp[water_temp_non_nan_idx]

occupancy_rna = numpy.sum((rel_s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
occupancy_dna = numpy.sum((rel_s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)

subset_idx = (occupancy_rna==1) & (occupancy_dna==1)

rel_s_by_s_rna_subset = rel_s_by_s_rna[subset_idx,:]
rel_s_by_s_dna_subset = rel_s_by_s_dna[subset_idx,:]




slope_range_all = []
slope_linear_rna_all = []
slope_linear_dna_all = []
slope_linear_ratio_all = []

mean_rna_all = []
mean_dna_all = []
for i_idx in range(sum(subset_idx)):

    afd_rna_i = rel_s_by_s_rna_subset[i_idx,:]
    afd_dna_i = rel_s_by_s_dna_subset[i_idx,:]

    mean_rna_i = numpy.mean(afd_rna_i)
    mean_dna_i = numpy.mean(afd_dna_i)

    #afd_ratio_i = afd_rna_i/afd_dna_i
    slope_linear_all = []
    for afd_i_k in [afd_rna_i, afd_dna_i, afd_rna_i/afd_dna_i]:

        # Fourier transform
        frq, f = signal.periodogram(afd_i_k)

        frq = numpy.log10(frq[1:])  # cut zero -> log(0) = -INF
        f = numpy.log10(numpy.abs(f.astype(complex))[1:])

        frq = frq[~numpy.isnan(f)]
        f = f[~numpy.isnan(f)]

        frq = frq[numpy.isfinite(f)]
        f = f[numpy.isfinite(f)]

        if len(frq) > 5:
            #p_spline = get_natural_cubic_spline_model(frq, f, minval=min(frq), maxval=max(frq), n_knots=4)

            #y = p_spline.predict(frq)

            #deriv = (y[1:] - y[:-1]) / (frq[1:] - frq[:-1])
            #slope_spline = min(deriv)

            # only consider frequencies which correspond to periods that are smaller than (length_timeseries/10)
            # otherwise effects from windowing
            f = f[frq >= min(frq) + 1]
            frq = frq[frq >= min(frq) + 1]

            # linear fit
            p_lin, cov = numpy.polyfit(frq, f, deg=1, cov=True)

            slope_linear = p_lin[0]
            std_slope_linear = numpy.sqrt(cov[0, 0])

            slope_linear_all.append(slope_linear)


    if len(slope_linear_all) >= 2:

        slope_range_all.extend(slope_linear_all)

        slope_linear_rna_all.append(slope_linear_all[0])
        slope_linear_dna_all.append(slope_linear_all[1])

        slope_linear_ratio_all.append(slope_linear_all[2])

        mean_rna_all.append(mean_rna_i)
        mean_dna_all.append(mean_dna_i)

        # DNA, RNA
        
print(numpy.mean(slope_linear_ratio_all), numpy.std(slope_linear_ratio_all))


fig, ax = plt.subplots(figsize=(4,4))

min_data, max_data = min(slope_range_all), max(slope_range_all)
ax.scatter(slope_linear_dna_all, slope_linear_rna_all, s=20, alpha=0.8, c='k', zorder=2)
ax.plot([min_data, max_data], [min_data, max_data], ls=':', lw='2', c='k', zorder=1)

slope, intercept, r_value, p_value, std_err = stats.linregress(slope_linear_dna_all, slope_linear_rna_all)

#print(slope)

x_range =  numpy.linspace(min_data, max_data, 10000)
y_fit_range = (slope*x_range + intercept)

ax.plot(x_range, y_fit_range, c='k', lw=2.5, linestyle='--', zorder=3, label="OLS regression slope")

#ax.scatter(distance_all, rho_all, zorder=2, alpha=0.3)
ax.set_xlim([min_data, max_data])
ax.set_ylim([min_data, max_data])
ax.set_xlabel("Power spectral density slope, DNA", fontsize = 11)
ax.set_ylabel("Power spectral density slope, RNA", fontsize = 11)

fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%sspectral_density_slope.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
   


# mean vs. slope

fig, ax = plt.subplots(figsize=(4,4))

slope_dna, intercept_dna, r_value_dna, p_value_dna, std_err_dna = stats.linregress(numpy.log10(mean_dna_all), slope_linear_dna_all)
slope_rna, intercept_rna, r_value_rna, p_value_rna, std_err_rna = stats.linregress(numpy.log10(mean_rna_all), slope_linear_rna_all)

#print(slope_dna, slope_rna)
#print(p_value_dna, p_value_rna)

ax.scatter(mean_dna_all, slope_linear_dna_all, s=20, alpha=0.8, c='k', zorder=2)
ax.scatter(mean_rna_all, slope_linear_rna_all, s=20, alpha=0.8, zorder=2, facecolors='white', edgecolors='k')
ax.set_xscale('log', basex=10)

fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%smean_vs_spectral_slope.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

#fig = plt.figure(figsize = (10, 6))
#fig.subplots_adjust(bottom= 0.15)

#ax_dna = plt.subplot2grid((2, 1), (0, 0), colspan=1)
#ax_rna = plt.subplot2grid((2, 1), (1, 0), colspan=1)

#min_data, max_data = min(slope_range_all), max(slope_range_all)
#ax_dna.scatter(mean_dna_all, slope_linear_dna_all, s=20, alpha=0.8, c='k', zorder=2)
#ax.plot([min_data, max_data], [min_data, max_data], ls=':', lw='2', c='k', zorder=1)
#ax.scatter(distance_all, rho_all, zorder=2, alpha=0.3)
#ax.set_xlim([min_data, max_data])
#ax.set_ylim([min_data, max_data])
#ax_dna.set_xlabel("Mean relative abundance DNA", fontsize = 11)
#ax_dna.set_ylabel("Power spectral density slope, DNA", fontsize = 11)
#




