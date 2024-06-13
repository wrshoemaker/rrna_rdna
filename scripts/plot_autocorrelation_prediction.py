import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats, signal





s_by_s, otu_labels, samples = utils.load_count_data()
to_keep_idx = otu_labels != 'Otu000001'
s_by_s = s_by_s[to_keep_idx,:]
otu_labels = otu_labels[to_keep_idx]
#print(s_by_s.shape)

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


days = numpy.asarray([metadata_dict[s]['day'] for s in sample_type_rna])

mean_rna_all = []
mean_dna_all = []

fig, ax = plt.subplots(figsize=(4,4))
for i_idx in range(sum(subset_idx)):

    afd_rna_i = rel_s_by_s_rna_subset[i_idx,:]
    afd_dna_i = rel_s_by_s_dna_subset[i_idx,:]
    afd_dna_log_i = numpy.log10(afd_dna_i)

    afd_ratio_i = afd_rna_i/afd_dna_i

    afd_ratio_i_all = []
    log_growth_dna_all = []
    afd_dna_i_all = []

    for t in range(len(afd_ratio_i)-1):

        afd_ratio_i_t = afd_ratio_i[t]
        # per-day change
        log_growth_dna = (afd_dna_log_i[t+1] - afd_dna_log_i[t])/(days[t+1] - days[t])
        
        afd_ratio_i_all.append(afd_ratio_i_t)
        log_growth_dna_all.append(log_growth_dna)
        afd_dna_i_all.append(afd_dna_i[t+1])

    ax.scatter(afd_ratio_i_all, afd_dna_i_all, alpha=0.4, s=0.4)

    afd_ratio_i_all = numpy.asarray(afd_ratio_i_all)
    #log_growth_dna_all = numpy.asarray(log_growth_dna_all)
    afd_dna_i_all = numpy.asarray(afd_dna_i_all)

    log_afd_ratio_i_all = numpy.log10(afd_ratio_i_all)
    log_afd_dna_i_all = numpy.log10(afd_dna_i_all)
    #slope, intercept, r_value, p_value, std_err = stats.linregress(log_afd_ratio_i_all, log_growth_dna_all)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_afd_ratio_i_all, log_afd_dna_i_all)

    #merged_ = numpy.concatenate([log_afd_ratio_i_all, log_growth_dna_all])
    merged_ = numpy.concatenate([log_afd_ratio_i_all, afd_dna_i_all])
    x_range =  numpy.linspace(min(merged_) , max(merged_) , 10000)
    y_fit_range = slope*x_range + intercept
    #ax.plot(10**x_range, y_fit_range, lw=1.5, ls='--', zorder=1)
    #ax.plot(10**x_range, 10**y_fit_range, lw=1.5, ls='--', zorder=1)

    print(slope)



ax.set_xscale('log', basex=10)
ax.set_yscale('log', basey=10)
ax.set_xlabel("RNA/DNA ratio, t", fontsize=10)
#ax.set_ylabel('Log ratio of DNA abundance\nbetween adjacent timepoints', fontsize=10)
ax.set_ylabel('DNA at t+1', fontsize=10)


fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%sautocorrelation_no_otu1.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



prediction_dict = {}


