import config
import numpy
import pickle
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.gridspec as gridspec

import sine_parameter_utils

from scipy import stats

taxonomic_level = 'genus'
taxonomy_dict = utils.build_taxonomy_dict()
rrna_copy_dict = utils.make_rrna_copy_dict()
rrna_copy_taxa = numpy.asarray(list(rrna_copy_dict.keys()))

param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))

genus_param = [taxonomy_dict[k][taxonomic_level] for k in param_dict['otu_labels']]

#days = 
s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
s_by_s_dna = s_by_s[:,(sample_type=='DNA')]


s_by_s_dna_to_keep = []

to_keep_idx = []
rrna_copy_number = []
mcn = numpy.zeros(len(param_dict['data']['clr_afd']['DNA'][0]))
n_otus = 0
for g_idx, g in enumerate(genus_param):

    if g in rrna_copy_dict:

        #print(g, param_dict['otu_labels'][g_idx])

        s_by_s_idx = numpy.where(otu_labels == param_dict['otu_labels'][g_idx])[0][0]
        afd_g = s_by_s_dna[s_by_s_idx,:]
        copy_number_g = afd_g*rrna_copy_dict[g]
        s_by_s_dna_to_keep.append(copy_number_g)
        


from scipy.stats import gmean

s_by_s_dna_to_keep = numpy.vstack(s_by_s_dna_to_keep)

# length of vector is # of samples
n_reads_s_by_s_dna_to_keep = gmean(s_by_s_dna_to_keep, axis=0)
cmr_s_by_s_dna_to_keep = (numpy.log(s_by_s_dna_to_keep) - numpy.log(n_reads_s_by_s_dna_to_keep))
print(numpy.sum(cmr_s_by_s_dna_to_keep, axis=0))


mcn = mcn/n_otus



fig = plt.figure(figsize = (4.5, 4)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=1, ncols=1)

ax = fig.add_subplot(gs[0, 0])


ax.plot(param_dict['data']['days']['DNA'][0], mcn, color='k', zorder=2)


fig.subplots_adjust(hspace=0.4, wspace=0.3)
fig_name = "%stime_vs_mcn.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()



#for i_idx in to_keep_idx:
