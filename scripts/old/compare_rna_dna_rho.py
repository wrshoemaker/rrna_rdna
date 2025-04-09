import config
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats, signal


s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

min_occupancy = 1/123

s_by_s_dna, s_by_s_rna, otu_labels_subset = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1)

rel_s_by_s_rescaled_dna = utils.rescale_s_by_s(s_by_s_dna)
rel_s_by_s_rescaled_rna = utils.rescale_s_by_s(s_by_s_rna)


for i_idx in range(rel_s_by_s_rescaled_dna.shape[0]):

    afd_dna_i = rel_s_by_s_rescaled_dna[i_idx,:]
    afd_rna_i = rel_s_by_s_rescaled_rna[i_idx,:]

    #if numpy.all(afd_dna_i == 0) or numpy.all(afd_rna_i == 0):
    #    continue

    rho_i = numpy.corrcoef(afd_dna_i, afd_rna_i)[0,1]

    print(rho_i)


