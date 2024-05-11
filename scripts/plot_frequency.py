import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats, signal


from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# https://github.com/lanadescheemaeker/logistic_models/blob/master/noise_analysis.py


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

occupancy_rna = numpy.sum((rel_s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
occupancy_dna = numpy.sum((rel_s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)

subset_idx = (occupancy_rna==1) & (occupancy_dna==1)

rel_s_by_s_rna_subset = rel_s_by_s_rna[subset_idx,:]
rel_s_by_s_dna_subset = rel_s_by_s_dna[subset_idx,:]

mad_rna_subset = numpy.mean(rel_s_by_s_rna_subset, axis=1)
mad_dna_subset = numpy.mean(rel_s_by_s_dna_subset, axis=1)
otu_labels_subset = otu_labels[subset_idx]

rescaled_rel_s_by_s_rna_subset = (rel_s_by_s_rna_subset.T/mad_rna_subset).T
rescaled_rel_s_by_s_dna_subset = (rel_s_by_s_dna_subset.T/mad_dna_subset).T

rescaled_rel_s_by_s_ratio_i = rescaled_rel_s_by_s_rna_subset/rescaled_rel_s_by_s_dna_subset

#mad_ratio_i = mad_rna_i/mad_dna_i
# len(mad_rna_subset) = 25
# 5x5 plot

fig = plt.figure(figsize = (20, 20))
fig.subplots_adjust(bottom= 0.15)

idx_all = list(range(len(mad_rna_subset)))
chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]


for chunk_idx, chunk in enumerate(chunk_all):

    for c_idx, c in enumerate(chunk):

        ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))
        
        rescaled_afd_dna_i = rescaled_rel_s_by_s_dna_subset[c,:]
        rescaled_afd_rna_i = rescaled_rel_s_by_s_rna_subset[c,:]
        rescaled_afd_ratio_i = rescaled_rel_s_by_s_ratio_i[c,:]
        
        # [DNA, RNA, ratio]
        types = ['DNA', 'RNA', 'ratio']
        for afd_i_idx, afd_i in enumerate([rescaled_afd_dna_i, rescaled_afd_rna_i, rescaled_afd_ratio_i]):
        
            # Fourier transform
            # frq = sample frequencies
            #f = power spectral density or power spectrum of x.
            frq, f = signal.periodogram(afd_i)
            frq = frq.astype(float)

            frq = numpy.log10(frq[1:])  # cut zero -> log(0) = -INF
            f = numpy.log10(numpy.abs(f.astype(complex))[1:])

            # remove nans
            frq = frq[~numpy.isnan(f)]
            f = f[~numpy.isnan(f)]

            # make sure numbers are finite
            frq = frq[numpy.isfinite(f)]
            f = f[numpy.isfinite(f)]

            ax.plot(frq, f, c=utils.dna_rna_color_dict[types[afd_i_idx]], ls='-', alpha=0.8, label=utils.rescaled_label_dict[types[afd_i_idx]])

        ax.set_title(otu_labels_subset[c], fontsize=11)
        ax.axvline(x=numpy.log10(2*numpy.pi/365), label=r'$\mathrm{log}_{10} \frac{2 \pi}{365}$', ls=':', c='k', zorder=2)
        ax.set_xlabel("Sample frequency, log10", fontsize=10)
        ax.set_ylabel("Power spectrum, log10", fontsize=10)

        if c == 0:
            ax.legend(loc='lower left')



fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%sfrequency.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()


    #print(frq, f)


    #print(len(frq), len(rescaled_afd_ratio_i))

    

