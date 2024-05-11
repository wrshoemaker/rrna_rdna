import config
import utils
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm


rrna_copy_dict = utils.make_rrna_copy_dict()
metadata_dict = utils.build_metadata_dict()
s_by_s, otu_labels, samples = utils.load_count_data()


def calculate_mean_copy_number(dna_or_rna):

    # only  use dna samples
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    sample_type_dna_idx = (sample_type==dna_or_rna)
    samples_dna = samples[sample_type_dna_idx]

    #samples_dna = samples[sample_type_dna_idx]
    s_by_s_dna = s_by_s[:,sample_type_dna_idx]
    otu_to_keep_idx = numpy.sum((s_by_s_dna>0), axis=1) > 0
    s_by_s_dna = s_by_s_dna[otu_to_keep_idx,:]
    otu_labels_dna = otu_labels[otu_to_keep_idx]


    # coarse-grain DNA samples
    s_by_s_coarse, coarse_labels, n_coarse, taxa_to_keep_idx = utils.coarse_grain_abundances_by_taxonomy(s_by_s_dna, otu_labels_dna)

    #coarse_labels_clean = []
    coarse_labels_to_keep = []
    copy_number_sorted = []
    for c in coarse_labels:

        # split candidatus
        if 'Candidatus' in c:
            #print(c)
            c_new = ' '.join(c.split('_'))
        else:
            c_new = c

        
        # check in dictionary
        if c_new in rrna_copy_dict:
        
        
            coarse_labels_to_keep.append(c)
            copy_number_sorted.append(rrna_copy_dict[c_new])



    #coarse_labels_clean = numpy.asarray(coarse_labels_clean)

    #coarse_labels_set = set(coarse_labels_clean.tolist())
    #rrna_taxa_set = set(rrna_copy_dict.keys())

    #intersection = coarse_labels_set & rrna_taxa_set
    #union = coarse_labels_set | rrna_taxa_set



    #coarse_labels_to_keep = numpy.asarray(list(intersection))

    coarse_labels_to_keep_idx = numpy.asarray([numpy.where(coarse_labels == c)[0][0] for c in coarse_labels_to_keep])


    #rel_s_by_s_coarse = s_by_s_coarse/numpy.sum(s_by_s_coarse, axis=0)
    #rel_s_by_s_coarse_to_keep = rel_s_by_s_coarse[coarse_labels_to_keep_idx,:]
    #rel_s_by_s_coarse_to_keep = rel_s_by_s_coarse_to_keep/numpy.sum(rel_s_by_s_coarse_to_keep, axis=0)
    #rel_s_by_s_copy_number = rel_s_by_s_coarse_to_keep

    s_by_s_coarse_to_keep = s_by_s_coarse[coarse_labels_to_keep_idx,:]


    #mad = numpy.mean(rel_s_by_s_coarse, axis=1)
    #mad_included = mad[coarse_labels_to_keep_idx]
    #mad_excluded = numpy.delete(mad, coarse_labels_to_keep_idx)


    copy_number_sorted = numpy.asarray(copy_number_sorted)

    copy_number_diag = numpy.diag(copy_number_sorted) # Create a diagonal matrix
    #s_by_s_copy_number = copy_number_diag @ s_by_s_coarse_to_keep
    s_by_s_copy_number = numpy.divide(s_by_s_coarse_to_keep.T, copy_number_sorted).T

    mean_copy_number = numpy.sum(s_by_s_coarse_to_keep, axis=0)/numpy.sum(s_by_s_copy_number, axis=0)

    days = numpy.asarray([metadata_dict[s]['day'] for s in samples_dna])

    return samples_dna, days, mean_copy_number



samples_dna, days_dna, mcn_dna = calculate_mean_copy_number('DNA')
samples_rna, days_rna, mcn_rna = calculate_mean_copy_number('RNA')


mcn_ratio = mcn_rna/mcn_dna



#temp_dna_filter = temp_dna_sort_idx


# days vs. MCN
fig = plt.figure(figsize = (10, 6))
fig.subplots_adjust(bottom= 0.15)

ax_dna = plt.subplot2grid((3, 1), (0, 0), colspan=1)
ax_rna = plt.subplot2grid((3, 1), (1, 0), colspan=1)
ax_ratio = plt.subplot2grid((3, 1), (2, 0), colspan=1)

# CV of log ratio
ax_dna.plot(days_dna, mcn_dna, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
ax_dna.scatter(days_dna, mcn_dna, s=5, alpha=0.8, c='k', zorder=1, label='DNA')

ax_rna.plot(days_rna, mcn_rna, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
#ax_rna.scatter(days_rna, mcn_rna, s=20, alpha=0.8, zorder=2, facecolors='white', edgecolors='k', label='RNA')
ax_rna.scatter(days_dna, mcn_rna, s=5, alpha=0.8, c='k', zorder=1)

ax_ratio.plot(days_rna, mcn_ratio, lw=1, ls='-', alpha=0.5, c='k', zorder=1)
#ax_ratio.scatter(days_rna, mcn_ratio, s=20, alpha=0.8, zorder=2, facecolors='white', edgecolors='k', label='RNA')
ax_ratio.scatter(days_dna, mcn_ratio, s=5, alpha=0.8, c='k', zorder=1)


#ax.ax_time(distance_all, rho_all, zorder=2, alpha=0.3)
ax_dna.set_xlabel("Time (days)", fontsize = 9)
ax_rna.set_xlabel("Time (days)", fontsize = 9)
ax_ratio.set_xlabel("Time (days)", fontsize = 9)

ax_dna.set_ylabel("Mean rRNA operon\ncopy number, DNA", fontsize = 7)
ax_rna.set_ylabel("Mean rRNA operon\ncopy number, RNA", fontsize = 7)
ax_ratio.set_ylabel("Ratio of RNA and DNA mean\nrRNA operon copy number", fontsize = 7)




fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%stime_vs_mcn.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()

