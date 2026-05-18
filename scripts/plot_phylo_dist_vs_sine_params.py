import config
import sys
import argparse
import copy
import numpy
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
import matplotlib.gridspec as gridspec

from scipy import stats, signal
from itertools import combinations


# numdifftools also installed
import pickle
#import ete4

import utils
import sine_parameter_utils


dist_dict_path = config.data_directory + 'otu_dist_dict.pickle'
#otu_list = ['Otu000001', 'Otu000002', 'Otu000003', 'Otu000004', 'Otu000008', 'Otu000009', 'Otu000014', 'Otu000016', 'Otu000019', 'Otu000021', 'Otu000023', 'Otu000024', 'Otu000028', 'Otu000030', 'Otu000032', 'Otu000034', 'Otu000037', 'Otu000041', 'Otu000046', 'Otu000050', 'Otu000051', 'Otu000058', 'Otu000075', 'Otu000093', 'Otu000131']



def build_phylo_dist_dict():

    dist_dict = {}
    tree_path = "%sasv_w_outgroup_aligned_clean.fna.raxml.bestTree" % config.data_directory
    tree = ete4.Tree(tree_path)
    # OTUs on tree label have five digits, e.g., Otu54350
    otu_all = [str(s) for s in tree.leaf_names() if str(s) != 'NC_005042_1_353331_354795_Prochlorococcus_marinus_subsp_marinus_str_CCMP1375_complete_genome']
    otu_pair_all = list(combinations(otu_all, 2))

    for otu_pair in otu_pair_all:

        otu_1 = str(otu_pair[0])
        otu_2 = str(otu_pair[1])
        otu_pair_dist = tree.get_distance(otu_1, otu_2)
        dist_dict[otu_pair] = otu_pair_dist

    sys.stderr.write("Saving distance dictionary...\n")
    with open(dist_dict_path, 'wb') as outfile:
        pickle.dump(dist_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)




def plot_dist_vs_sine_parameters():

    param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
    dist_dict = pickle.load(open(dist_dict_path, "rb"))
    otu_pair_all = list(dist_dict.keys())
    
    fig = plt.figure(figsize = (8.5, 12)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=3, ncols=2)
    for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

        for param_idx, param in enumerate(['amp', 'freq', 'phase']):

            ax = fig.add_subplot(gs[param_idx, data_type_idx])

            if param_idx == 0:
                ax.set_title(utils.rescaled_label_clr_dict[data_type], fontsize=14, color=utils.dna_rna_color_dict[data_type], fontweight='bold')

               
            dist_all = []
            param_delta_all = []
            for otu_pair in otu_pair_all:

                dist_all.append(dist_dict[otu_pair])
                param_delta_all.append(param_dict['%s_mle' % param][data_type][param_dict['otu_labels'].index(otu_pair[0])] - param_dict['%s_mle' % param][data_type][param_dict['otu_labels'].index(otu_pair[1])])


            param_delta_all = numpy.absolute(param_delta_all)

            ax.scatter(dist_all, param_delta_all, alpha=0.7, s=10, color=utils.dna_rna_color_dict[data_type], zorder=1)

            ax.set_xlabel('Phylogenetic distance between ASVs, ' + r'$d_{i,j}$', fontsize=11)
            
            if param == 'amp':
                label = 'Abs. diff. of amplitidues, ' + r'$ | A_{i} - A_{j} |$'

            elif param == 'freq':
                label = 'Abs. diff. of oscillation timescales, ' + r'$ | \tau_{\mathrm{env}}^{(i)} -  \tau_{\mathrm{env}}^{(j)} |$'

            else:
                label = 'Abs. diff. of phases timescales, ' + r'$ | \psi_{i} -  \psi_{j} |$'


            ax.set_ylabel(label, fontsize=10)



    fig.subplots_adjust(hspace=0.3, wspace=0.35)
    fig_name = "%sphylo_dist_vs_sine_params.png" % config.analysis_directory
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()








if __name__ == "__main__":

    print("Running distance-decay analysis...")

    #build_phylo_dist_dict()
    plot_dist_vs_sine_parameters()

