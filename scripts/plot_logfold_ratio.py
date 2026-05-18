import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats, signal
import pickle
import sine_parameter_utils


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

taxonomy_dict = utils.build_taxonomy_dict()

s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=0)

occupancy_dna = numpy.sum((s_by_s_dna>0), axis=1)/len(days)
occupancy_rna = numpy.sum((s_by_s_rna>0), axis=1)/len(days)

occupancy_idx = (occupancy_rna>=1) & (occupancy_dna>=1)
otu_labels = otu_labels[occupancy_idx]

param_dict =  pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, 'rb'))
otu_labels_param = param_dict['otu_labels']

s_by_s_dna = s_by_s_dna[occupancy_idx,:]
s_by_s_rna = s_by_s_rna[occupancy_idx,:]

rel_s_by_s_dna = s_by_s_dna/numpy.sum(s_by_s_dna, axis=0)
rel_s_by_s_rna = s_by_s_rna/numpy.sum(s_by_s_rna, axis=0)

log_ratio_dna = numpy.log10(rel_s_by_s_dna[:,1:]) - numpy.log10(rel_s_by_s_dna[:,:-1])
log_ratio_rna = numpy.log10(rel_s_by_s_rna[:,1:]) - numpy.log10(rel_s_by_s_rna[:,:-1])

delta_t = days[1:] - days[:-1]




idx_all = list(range(len(otu_labels)))
chunk_all = [idx_all[x:x+5] for x in range(0, len(idx_all), 5)]

fig = plt.figure(figsize = (20, 20))
fig.subplots_adjust(bottom= 0.15)

asv_count = 0
for chunk_idx, chunk in enumerate(chunk_all):

    for c_idx, c in enumerate(chunk):
        
        ax = plt.subplot2grid((len(chunk_all), len(chunk_all[0])), (chunk_idx, c_idx))

        otu_label_c = otu_labels_param[asv_count]

        idx_c = numpy.where(otu_labels == otu_label_c)[0]

        log_ratio_dna_i = log_ratio_dna[idx_c,:]/delta_t
        log_ratio_rna_i = log_ratio_rna[idx_c,:]/delta_t

        hist_afd_dna, bins_afd_dna = utils.get_hist_and_bins(log_ratio_dna_i, bins=12)
        hist_afd_rna, bins_afd_rna = utils.get_hist_and_bins(log_ratio_rna_i, bins=12)

        max_logfold = numpy.max(numpy.absolute(numpy.concatenate([log_ratio_dna_i, log_ratio_rna_i])))

        #ax.scatter(bins_afd_dna, hist_afd_dna, s=7, color=utils.dna_rna_color_dict['DNA'], alpha=0.7, lw=1)
        #ax.scatter(bins_afd_rna, hist_afd_rna, s=7, color=utils.dna_rna_color_dict['RNA'], alpha=0.7, lw=1)

        ax.hist(log_ratio_dna_i[0], bins=12, density=True, histtype='step', alpha=1, lw=3, color=utils.dna_rna_color_dict['DNA'], zorder=1, label='rDNA')
        ax.hist(log_ratio_rna_i[0], bins=12, density=True, histtype='step', alpha=1, lw=3, color=utils.dna_rna_color_dict['RNA'], zorder=1, label='rRNA')

        ax.axvline(x=0, ls=':', c='k', lw=2, zorder=2, label='Stationarity')
        ax.set_yscale('log', base=10)
        ax.set_xlabel("Log-fold change in " + r'$\hat{x}_{i}(t)$' +  '\nb/w weekly samples, ' + r'$\Delta \ell$', fontsize=10)
        ax.set_ylabel("Probability density", fontsize=12)
        ax.set_xlim([-1*max_logfold, max_logfold])
        #ax.set_title(otu_labels[c], fontsize=12)

        ax.set_title('ASV %d (%s)' % (asv_count+1, taxonomy_dict[otu_label_c]['family']), fontsize=11)


        if c == 0:
            ax.legend(loc='upper left', fontsize=10)


        asv_count += 1



fig.subplots_adjust(hspace=0.4, wspace=0.40)
fig_name = "%slogfold_ratio.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()




