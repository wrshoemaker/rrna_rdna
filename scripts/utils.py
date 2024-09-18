
import config 
from datetime import datetime
import numpy
#import pandas
import math
import sympy
from scipy.stats import gmean

from matplotlib import cm


from collections import Counter



# colors_dict = {'0':'#87CEEB', '1': '#FFA500', '2':'#FF6347'}

dna_rna_color_dict = {'RNA': '#FF6347', 'DNA': '#87CEEB', 'ratio':'k'}

color_radius = 2

rescaled_label_dict = {'RNA':'Rescaled RNA, ' + r'$r_{i}(t)$', 'DNA': 'Rescaled DNA, ' + r'$d_{i}(i)$', 'ratio': 'Rescaled RNA:DNA, ' + r'$\phi_{i}(t)$'}
#rescaled_label_dict = {'RNA':'Rescaled RNA, ' + r'$r_{i}(t)$', 'DNA': 'Rescaled DNA, ' + r'$d_{i}(i)$', 'ratio': 'Rescaled RNA:DNA, ' + r'$\phi_{i}(t)$'}
rescaled_label_clr_dict = {'RNA':'RNA', 'DNA': 'DNA', 'ratio': 'RNA - DNA'}

sample_label_dict = {'RNA': 'RNA', 'DNA':'DNA', 'ratio': 'RNA-DNA ratio'}

data_type_all = ['DNA', 'RNA', 'ratio']
env_variables_all = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph']


taxonomic_ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']


env_variable_label_dict = {'water_temp': 'Water temperature (°C)', 'specific_conductivity': 'Specific conductivity (mS/cm)', 
                            'dissolved_oxygen': 'Dissolved oxygen (mg/L)', 'salinity': 'Salinity (PSS)',
                            'secchi_depth': 'Secchi depth (m)', 'ph':'pH', 'air_temperature': 'Air temperature (°C)'}


env_variable_no_unit_label_dict = {'water_temp': 'Water temperature', 'specific_conductivity': 'Specific conductivity', 
                            'dissolved_oxygen': 'Dissolved oxygen', 'salinity': 'Salinity',
                            'secchi_depth': 'Secchi depth', 'ph':'pH', 'air_temperature': 'Air temperature'}


def get_p_value_latex_label_dict(p_value):

    if p_value <= 0.05:
        label = r'$P \leq 0.05$'


    else:
        label = r'$P \nleq 0.05$'

    return label




def make_colormap(sample_type, n_obs, lower_linspace_bound=0.1):

    if sample_type == 'DNA':
        color_ = 'Blues'

    elif sample_type == 'RNA':
        color_ = 'Reds'

    else:
        print("Argument not recognized")


    color_range =  numpy.linspace(lower_linspace_bound, 1.0, n_obs)
    rgb_ = cm.get_cmap(color_)( color_range )

    return rgb_



def build_metadata_dict():

    file_ = open('%sdesign.csv' % config.data_directory, 'r')
    file_header = file_.readline()
    
    metadata_dict = {}

    for line in file_:

        line = line.strip().split(',')

        sample = line[0].strip('"')
        sample_type = line[1].strip('"')

        metadata_dict[sample] = {}
        metadata_dict[sample]['sample_type'] = sample_type

    file_.close()

    # environmental metadata
    file_environment = open('%sUnivLakeSurface.txt' % config.data_directory, 'r')
    file_environment_header = file_environment.readline()
    file_environment_header = file_environment_header.strip().split('\t')
    
    for line in file_environment:

        line = line.strip().split('\t')
        
        #print(line)
        sample = line[0]
        if sample == '033':
            continue
        
        date, depth, water_temp, specific_conductivity, dissolved_oxygen, salinity, secchi_depth, ph, air_temperature  = line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[10]

        if depth == 'NA':
            depth = numpy.nan
        else:
            depth = float(depth)

        if water_temp == 'NA':
            water_temp = numpy.nan
        else:
            water_temp = float(water_temp)

        if specific_conductivity == 'NA':
            specific_conductivity = numpy.nan
        else:
            specific_conductivity = float(specific_conductivity)

        if dissolved_oxygen == 'NA':
            dissolved_oxygen = numpy.nan
        else:
            dissolved_oxygen = float(dissolved_oxygen)

        if salinity == 'NA':
            salinity = numpy.nan
        else:
            salinity = float(salinity)

        if secchi_depth == 'NA':
            secchi_depth = numpy.nan
        else:
            secchi_depth = float(secchi_depth)

        if ph == 'NA':
            ph = numpy.nan
        else:
            ph = float(ph)

        if (air_temperature == 'NA') or (air_temperature == ''):
            air_temperature = numpy.nan
        else:
            air_temperature = float(air_temperature)


        if date == '04/04014':
            date = '04/04/14'

        if date == '07/26/19':
            date = '07/26/13'

        if date == '02/06/25':
            date = '02/06/15'

        datetime_object = datetime.strptime(date, '%m/%d/%y')

        if sample == '001':
            days = 0

        else:
            days = (datetime_object - metadata_dict['ULc001']['date']).days

        rna_sample = 'ULc' + sample
        dna_sample = 'ULD' + sample

        for sample_i in [rna_sample, dna_sample]:

            metadata_dict[sample_i]['date'] = datetime_object
            metadata_dict[sample_i]['day'] = days
            metadata_dict[sample_i]['depth'] = depth
            metadata_dict[sample_i]['water_temp'] = water_temp
            metadata_dict[sample_i]['specific_conductivity'] = specific_conductivity
            metadata_dict[sample_i]['dissolved_oxygen'] = dissolved_oxygen
            metadata_dict[sample_i]['salinity'] = salinity
            metadata_dict[sample_i]['secchi_depth'] = secchi_depth
            metadata_dict[sample_i]['ph'] = ph
            metadata_dict[sample_i]['air_temperature'] = air_temperature

    
    file_environment.close()

    return metadata_dict


def build_taxonomy_dict():

    file_taxonomy = open('%sUL.bac.final.0.03.taxonomy' % config.data_directory, 'r')
    file_taxonomy_header = file_taxonomy.readline()
    
    taxonomy_dict = {}
    for line in file_taxonomy:

        line = line.strip().split('\t')

        taxonomy_dict[line[0]] = {}
        #for 

        taxa = line[2].strip().split(';')
        for t_idx, t in enumerate(taxa):
            t_clean = t.split('(')[0]

            if t_clean == '':
                t_clean = 'NA'

            taxonomy_dict[line[0]][taxonomic_ranks[t_idx]] = t_clean
    
    return taxonomy_dict


def load_count_data():

    file_counts = open('%sUL.bac.final.shared' % config.data_directory, 'r')
    file_counts_header = file_counts.readline().strip().split('\t')
    otu_labels = file_counts_header[3:]

    samples = []
    s_by_s = []
    for line in file_counts:
        line = line.strip().split('\t')

        samples.append(line[1])
        sad = line[3:]
        sad = [int(s) for s in sad]
        s_by_s.append(sad)


    s_by_s = numpy.asarray(s_by_s)
    otu_labels = numpy.asarray(otu_labels)
    samples = numpy.asarray(samples)

    # transpose
    s_by_s = s_by_s.T

    return s_by_s, otu_labels, samples
    


def make_rrna_copy_dict(level="genus", reformat_labels=False):
    level = level.lower()

    rna_file_path = '%srrnDB-5.8_pantaxa_stats_NCBI.tsv' % config.data_directory
    rna_file = open(rna_file_path, 'r')
    header = rna_file.readline()

    rrna_copy_dict = {}

    for line in rna_file:
        line_split = line.strip().split('\t')
        if line_split[1] == level:

            taxon = line_split[2]
            #median_rrna = line_split[7]
            mean_rrna = float(line_split[8])

            if reformat_labels == True:

                if 'Candidatus' in taxon:
                    taxon = ' '.join(taxon.split('_'))
      

            rrna_copy_dict[taxon] = mean_rrna

        

    rna_file.close()

    return rrna_copy_dict






def coarse_grain_abundances_by_taxonomy_old(count_array, asvs, taxonomic_level='genus'):

    # coarse grains get_counts() matrix by taxonomic label for a given taxonomic_level
    taxonomy_dict = build_taxonomy_dict()

    asvs_to_keep = list(taxonomy_dict.keys())
    taxon_labels_all = [taxonomy_dict[a][taxonomic_level] for a in asvs_to_keep if a in asvs]

    taxon_labels_all_array = numpy.asarray(taxon_labels_all)
    taxa_to_keep_idx = taxon_labels_all_array != "NA"

    #taxon_labels_all_clean = [t for t in taxon_labels_all if t != "NA"]
    taxon_labels_all_clean_array = taxon_labels_all_array[taxon_labels_all_array!="NA"]
    taxon_labels_all_clean = taxon_labels_all_clean_array.tolist()


    # return this list for null coarse-grained site-by-species matrix
    n_collapsed_nodes = list(dict(Counter(taxon_labels_all_clean)).values())

    # remove duplicates from taxon_labels_all_clean while keeping original order
    def unique(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    taxon_labels_all_clean_set = unique(taxon_labels_all_clean)
    taxon_labels_all_clean_set = numpy.asarray(taxon_labels_all_clean_set)
    taxon_labels_all_clean_copy = taxon_labels_all_clean.copy()

    to_remove_idx = [i for i, x in enumerate(taxon_labels_all) if x == "NA"]
    to_remove_idx = numpy.asarray(to_remove_idx)
    count_array_ = numpy.copy(count_array)

    print(count_array_.shape)

    # delete NA
    print(len(to_remove_idx))
    if len(to_remove_idx) > 0:
        count_array_ = numpy.delete(count_array_, to_remove_idx, axis=0)

    print(count_array_.shape)

    for taxon_i in taxon_labels_all_clean_set:

        taxon_i_idx = [i for i, x in enumerate(taxon_labels_all_clean_copy) if x == taxon_i]
        taxon_i_idx = numpy.asarray(taxon_i_idx)

        # remove taxon from the list
        taxon_labels_all_clean_copy = [s for s in taxon_labels_all_clean_copy if s != taxon_i]

        count_array_to_merge = count_array_[taxon_i_idx,]
        count_array_ = numpy.delete(count_array_, taxon_i_idx, axis=0)

        count_array_sum = numpy.sum(count_array_to_merge, axis=0)
        count_array_ = numpy.vstack((count_array_, count_array_sum))


    return count_array_, taxon_labels_all_clean_set, n_collapsed_nodes, taxa_to_keep_idx





def coarse_grain_abundances_by_taxonomy(count_array, asvs, taxonomic_level='genus'):

    # coarse grains get_counts() matrix by taxonomic label for a given taxonomic_level
    taxonomy_dict = build_taxonomy_dict()

    asvs_to_keep = list(taxonomy_dict.keys())
    #taxon_labels_all = [taxonomy_dict[a][taxonomic_level] for a in asvs_to_keep if a in asvs]
    taxon_labels_all = [taxonomy_dict[a][taxonomic_level] for a in asvs]

    taxa_to_keep_idx = numpy.asarray(taxon_labels_all)
    taxa_to_keep_idx = taxa_to_keep_idx != "NA"

    taxon_labels_all_clean = [t for t in taxon_labels_all if t != "NA"]

    # return this list for null coarse-grained site-by-species matrix
    n_collapsed_nodes = list(dict(Counter(taxon_labels_all_clean)).values())

    # remove duplicates from taxon_labels_all_clean while keeping original order
    def unique(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    taxon_labels_all_clean_set = unique(taxon_labels_all_clean)
    taxon_labels_all_clean_set = numpy.asarray(taxon_labels_all_clean_set)
    taxon_labels_all_clean_copy = taxon_labels_all_clean.copy()

    to_remove_idx = [i for i, x in enumerate(taxon_labels_all) if x == "NA"]
    to_remove_idx = numpy.asarray(to_remove_idx)
    count_array_ = numpy.copy(count_array)


    # delete NA
    if len(to_remove_idx) > 0:
        count_array_ = numpy.delete(count_array_, to_remove_idx, axis=0)


    for taxon_i in taxon_labels_all_clean_set:

        taxon_i_idx = [i for i, x in enumerate(taxon_labels_all_clean_copy) if x == taxon_i]
        taxon_i_idx = numpy.asarray(taxon_i_idx)

        # remove taxon from the list
        taxon_labels_all_clean_copy = [s for s in taxon_labels_all_clean_copy if s != taxon_i]

        count_array_to_merge = count_array_[taxon_i_idx,]
        count_array_ = numpy.delete(count_array_, taxon_i_idx, axis=0)

        count_array_sum = numpy.sum(count_array_to_merge, axis=0)
        count_array_ = numpy.vstack((count_array_, count_array_sum))


    return count_array_, taxon_labels_all_clean_set, n_collapsed_nodes, taxa_to_keep_idx




def get_hist_and_bins(flat_array, bins=20):

    # make sure its an array
    flat_array = numpy.asarray(flat_array)

    flat_array = flat_array[~numpy.isnan(flat_array)]

    # null is too large, so we are binning it for the plot in this script
    hist_, bin_edges_ = numpy.histogram(flat_array, density=True, bins=bins)
    bins_mean_ = numpy.asarray([0.5 * (bin_edges_[i] + bin_edges_[i+1]) for i in range(0, len(bin_edges_)-1 )])
    hist_to_plot = hist_[hist_>0]
    bins_mean_to_plot = bins_mean_[hist_>0]

    return hist_to_plot, bins_mean_to_plot



def calculate_shannon_diversity(sad):

    relative_sad = sad / sum(sad)
    relative_sad = relative_sad[relative_sad>0]
    shannon_diversity = -1*sum(relative_sad*numpy.log(relative_sad) )

    #shannon_diversity = -1*sum(relative_sad_i * numpy.log(relative_sad_i) for relative_sad_i in relative_sad)

    return shannon_diversity


def calculate_pielou_evenness(sad):

    return calculate_shannon_diversity(sad) / numpy.log(len(sad))



def calculate_sparsity(sad):

    return sum(sad==0)/len(sad)


def calculate_relative_richness(sad):

    return sum(sad>0)/len(sad)


def calculate_richness(sad):

    return sum(sad>0)



def rescale_log(array_):

    array_log = numpy.log10(array_)
    array_log_rescaled = (array_log - numpy.mean(array_log))/numpy.std(array_log)

    return array_log_rescaled





def predict_occupancy(s_by_s, species, totreads=numpy.asarray([])):

    # get squared inverse cv
    # assume that entries are read counts.
    rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

    beta_all = []
    mean_all = []

    for s in rel_s_by_s_np:

        var = numpy.var(s)
        mean = numpy.mean(s)

        beta = (mean**2)/var

        mean_all.append(mean)
        beta_all.append(beta)

    beta_all = numpy.asarray(beta_all)
    mean_all = numpy.asarray(mean_all)


    s_by_s_presence_absence = numpy.where(s_by_s > 0, 1, 0)

    occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]

    # calcualte total reads if no argument is passed
    # sloppy quick fix
    if len(totreads) == 0:
        totreads = s_by_s.sum(axis=0)

    # calculate mean and variance excluding zeros
    # tf = mean relative abundances
    tf = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]
        tf.append(numpy.mean(afd_no_zeros/ totreads[afd>0]))

    tf = numpy.asarray(tf)
    # go through and calculate the variance for each species

    tvpf_list = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]

        #N_reads = s_by_s.sum(axis=0)[numpy.nonzero(afd)[0]]
        tvpf_list.append(numpy.mean(  (afd_no_zeros**2 - afd_no_zeros) / (totreads[afd>0]**2) ))

    tvpf = numpy.asarray(tvpf_list)

    f = occupancies*tf
    vf= occupancies*tvpf

    # there's this command in Jacopo's code %>% mutate(vf = vf - f^2 )%>%
    # It's applied after f and vf are calculated, so I think I can use it
    # This should be equivalent to the mean and variance including zero
    vf = vf - (f**2)

    beta = (f**2)/vf
    theta = f/beta

    predicted_occupancies = []
    # each species has it's own beta and theta, which is used to calculate predicted occupancy
    for beta_i, theta_i in zip(beta,theta):
        predicted_occupancies.append(1 - numpy.mean( ((1+theta_i*totreads)**(-1*beta_i ))   ))

    predicted_occupancies = numpy.asarray(predicted_occupancies)

    species = numpy.asarray(species)
    rel_s_by_s = (s_by_s/s_by_s.sum(axis=0))
    mad = numpy.mean(rel_s_by_s, axis=1)



    return occupancies, predicted_occupancies, mad, beta, species




# https://github.com/weecology/macroecotools/blob/master/macroecotools/macroecotools.py
# code to cluster points
def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    unique_points = set([(x[i], y[i]) for i in range(len(x))])
    count_data = []
    logx, logy, logr = numpy.log10(x), numpy.log10(y), numpy.log10(radius)
    for a, b in unique_points:
        if logscale == 1:
            loga, logb = numpy.log10(a), numpy.log10(b)
            num_neighbors = len(x[((logx - loga) ** 2 +
                                   (logy - logb) ** 2) <= logr ** 2])
        else:
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data


def plot_color_by_pt_dens(x, y, radius, loglog=0):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = numpy.array(sorted(plot_data, key=lambda point: point[2]))

    return sorted_plot_data







def calculate_mean_copy_number(dna_or_rna='DNA', taxonomic_level='Genus'):

    rrna_copy_dict = make_rrna_copy_dict()
    metadata_dict = build_metadata_dict()
    s_by_s, otu_labels, samples = load_count_data()

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
    s_by_s_coarse, coarse_labels, n_coarse, taxa_to_keep_idx = coarse_grain_abundances_by_taxonomy(s_by_s_dna, otu_labels_dna, taxonomic_level)

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
            
            # save the coarse-grained label from the annoated OTUs
            # not the label from the rRNA copy number database
            coarse_labels_to_keep.append(c)
            copy_number_sorted.append(rrna_copy_dict[c_new])


    coarse_labels_to_keep_idx = numpy.asarray([numpy.where(coarse_labels == c)[0][0] for c in coarse_labels_to_keep])

    #rel_s_by_s_coarse = s_by_s_coarse/numpy.sum(s_by_s_coarse, axis=0)
    #rel_s_by_s_coarse_to_keep = rel_s_by_s_coarse[coarse_labels_to_keep_idx,:]
    #rel_s_by_s_coarse_to_keep = rel_s_by_s_coarse_to_keep/numpy.sum(rel_s_by_s_coarse_to_keep, axis=0)
    #rel_s_by_s_copy_number = rel_s_by_s_coarse_to_keep

    s_by_s_coarse_to_keep = s_by_s_coarse[coarse_labels_to_keep_idx,:]


    copy_number_sorted = numpy.asarray(copy_number_sorted)

    copy_number_diag = numpy.diag(copy_number_sorted) # Create a diagonal matrix
    #s_by_s_copy_number = copy_number_diag @ s_by_s_coarse_to_keep
    s_by_s_copy_number = numpy.divide(s_by_s_coarse_to_keep.T, copy_number_sorted).T

    mean_copy_number = numpy.sum(s_by_s_coarse_to_keep, axis=0)/numpy.sum(s_by_s_copy_number, axis=0)

    days = numpy.asarray([metadata_dict[s]['day'] for s in samples_dna])

    coarse_labels_to_keep = numpy.asarray(coarse_labels_to_keep)

    return samples_dna, days, coarse_labels_to_keep, mean_copy_number



def subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=1):

    # takes s_by_s, splits by RNA and DNA
    # calculates relative abundance
    # then selects samples with same occupancy in each dataset
    # returns (DNA, RNA) as relative abundance

    metadata_dict = build_metadata_dict()

    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

    sample_type_rna_idx = (sample_type=='RNA')
    sample_type_dna_idx = (sample_type=='DNA')

    #sample_type_rna = samples[sample_type_rna_idx]

    s_by_s_rna = s_by_s[:,sample_type_rna_idx]
    s_by_s_dna = s_by_s[:,sample_type_dna_idx]

    n_reads_rna = numpy.sum(s_by_s_rna, axis=0)
    n_reads_dna = numpy.sum(s_by_s_dna, axis=0)

    rel_s_by_s_rna = s_by_s_rna/n_reads_rna
    rel_s_by_s_dna = s_by_s_dna/n_reads_dna

    occupancy_rna = numpy.sum((s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
    occupancy_dna = numpy.sum((s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)

    occupancy_idx = (occupancy_rna>=min_occupancy) & (occupancy_dna>=min_occupancy)

    otu_labels_occupancy = otu_labels[occupancy_idx]
    
    rel_s_by_s_rna_occupancy = rel_s_by_s_rna[occupancy_idx,:]
    rel_s_by_s_dna_occupancy = rel_s_by_s_dna[occupancy_idx,:]

    return rel_s_by_s_dna_occupancy, rel_s_by_s_rna_occupancy, otu_labels_occupancy



def rescale_s_by_s(s_by_s):

    # rescale s_by_s by mean
    # resturn rescaled relative abundance matrix
    
    rel_s_by_s = s_by_s/numpy.sum(s_by_s, axis=0)
    mad = numpy.mean(rel_s_by_s, axis=1)
    rescaled_rel_s_by_s = (rel_s_by_s.T/mad).T

    return rescaled_rel_s_by_s



def clr_transform(s_by_s, otu_labels, samples, min_occupancy=1):

    # subset_n_reads=False

    # requires rel_s_by_s ASVs to all have occupancy = 1
    metadata_dict = build_metadata_dict()

    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])

    sample_type_rna_idx = (sample_type=='RNA')
    sample_type_dna_idx = (sample_type=='DNA')

    s_by_s_rna = s_by_s[:,sample_type_rna_idx]
    s_by_s_dna = s_by_s[:,sample_type_dna_idx]

    occupancy_rna = numpy.sum((s_by_s_rna>0), axis=1)/sum(sample_type_rna_idx)
    occupancy_dna = numpy.sum((s_by_s_dna>0), axis=1)/sum(sample_type_dna_idx)

    occupancy_idx = (occupancy_rna>=min_occupancy) & (occupancy_dna>=min_occupancy)

    otu_labels_occupancy = otu_labels[occupancy_idx]
    
    s_by_s_rna_occupancy = s_by_s_rna[occupancy_idx,:]
    s_by_s_dna_occupancy = s_by_s_dna[occupancy_idx,:]

    #n_reads_rna = numpy.sum(s_by_s_rna, axis=0)
    #n_reads_dna = numpy.sum(s_by_s_dna, axis=0)

    n_reads_rna_gmean = gmean(s_by_s_rna_occupancy, axis=0)
    n_reads_dna_gmean = gmean(s_by_s_dna_occupancy, axis=0)

    clr_s_by_s_rna = numpy.log(s_by_s_rna_occupancy/n_reads_rna_gmean)
    clr_s_by_s_dna = numpy.log(s_by_s_dna_occupancy/n_reads_dna_gmean)

    return clr_s_by_s_dna, clr_s_by_s_rna, otu_labels_occupancy




def clr_transform_sim(s_by_s, min_occupancy=1):

    # requires rel_s_by_s ASVs to all have occupancy = 1
    occupancy = numpy.sum((s_by_s>0), axis=1)/s_by_s.shape[1]

    occupancy_idx = (occupancy>=min_occupancy) #& (occupancy>=min_occupancy)
    
    s_by_s_occupancy = s_by_s[occupancy_idx,:]

    #n_reads_rna = numpy.sum(s_by_s_rna, axis=0)
    #n_reads_dna = numpy.sum(s_by_s_dna, axis=0)

    # geometric mean *across* species calculated for each sample
    # len(n_reads_gmean) = # samples
    n_reads_gmean = gmean(s_by_s_occupancy, axis=0)

    clr_s_by_s = numpy.log(s_by_s_occupancy/n_reads_gmean)

    return clr_s_by_s, occupancy_idx






def get_seasonal_tick_labels():

    # returns minor ticks, major ticks , and major tick labels

    # x axis in unit of days
    metadata_dict = build_metadata_dict()

    dna_samples = [k for k in metadata_dict.keys() if 'ULD' in k]
    days = [metadata_dict[k]['day'] for k in dna_samples]
    dates = [metadata_dict[k]['date'] for k in dna_samples]

    # sort tuples
    days_dates_tuple = list(zip(days, dates))
    days_dates_tuple.sort(key=lambda tup: tup[0])

    days = [k[0] for k in days_dates_tuple]
    dates = [k[1] for k in days_dates_tuple]

    total_days = days[-1]
    first_sample_date = dates[0]

    # number of times that a given date can appear in the time series
    # = floor[ (total #days - (difference between target date and start date) ) / 365 ]
    
    minor_days = []

    major_days = []
    major_labels = []


    for i in range(1, 13):

        # first of the month for every month
        diff_days_i = (datetime.strptime('%d/01/13' % i, '%m/%d/%y') - first_sample_date).days
        n_repeats = math.floor((total_days-diff_days_i)/365)
        #print(n_repeats, (total_days-diff_days_i)/365)
        
        if diff_days_i < 0:
            n_lower = 1
            n_upper = n_repeats+1

        else:
            n_lower = 0
            n_upper = n_repeats+1

        for n in range(n_lower, n_upper):
           
            days_i_n = diff_days_i + (365*n)
            minor_days.append(days_i_n)

            if i == 1:
                major_days.append(days_i_n)
                major_labels.append('Jan.')

            if i == 4:
                major_days.append(days_i_n)
                major_labels.append('Apr.')

            if i == 7:
                major_days.append(days_i_n)
                major_labels.append('Jul.')

            if i == 10:
                major_days.append(days_i_n)
                major_labels.append('Oct.')


    minor_days.sort()
    #print(total_days)

    major_tuple = list(zip(major_days, major_labels))
    major_tuple.sort(key=lambda tup: tup[0])

    major_days = [k[0] for k in major_tuple]
    major_labels = [k[1] for k in major_tuple]


    return minor_days, major_days, major_labels




def Klogn(cumK, c, mu0=-19, s0=5):
    # This function estimates the parameters (mu, s) of the lognormal distribution of K
    m1 = numpy.mean(numpy.log(cumK[cumK>c]))
    m2 = numpy.mean(numpy.log(cumK[cumK>c])**2)
    xmu = sympy.symbols('xmu')
    xs = sympy.symbols('xs')
    eq1 = -m1+xmu + sympy.sqrt(2/sympy.pi)*xs*sympy.exp(-((sympy.log(c)-xmu)**2)/2/(xs**2))/(sympy.erfc((sympy.log(c)-xmu)/sympy.sqrt(2)/xs))
    eq2 = -m2+xs**2+m1*xmu+sympy.log(c)*m1-xmu*sympy.log(c)

    sol = sympy.nsolve([eq1,eq2],[xmu,xs],[mu0,s0])

    return float(sol[0]), float(sol[1])



def estimate_k_and_sigma(s_by_s, min_occupancy=0.2):

    n_reads = s_by_s.sum(axis=0)

    rel_s_by_s = s_by_s/n_reads

    occupancy = numpy.sum(rel_s_by_s>0, axis=1)/len(n_reads)

    mediasq = numpy.mean(numpy.divide(s_by_s.T*(s_by_s.T - numpy.ones(numpy.shape(s_by_s.T))), (n_reads*(n_reads-1))[:,None]), axis=0 )  
    meanrelabd = numpy.mean(rel_s_by_s, axis=1) 


    temp = 1 + meanrelabd**2/(mediasq-meanrelabd**2)  
    sigma = numpy.where((temp!=0) & (~numpy.isnan(temp)), 2/temp, numpy.nan) 
    k = 2*meanrelabd/(2-sigma) 
    ids = numpy.nonzero((sigma>0) & (occupancy>min_occupancy)) 
    ids2 = numpy.nonzero(sigma>0)
    
    return k, sigma, ids, ids2, meanrelabd
   






def predict_mean_richness(s_by_s, species, totreads=numpy.asarray([])):

    # get squared inverse cv
    # assume that entries are read counts.
    rel_s_by_s_np = (s_by_s/s_by_s.sum(axis=0))

    beta_all = []
    mean_all = []

    for s in rel_s_by_s_np:

        var = numpy.var(s)
        mean = numpy.mean(s)

        beta = (mean**2)/var

        mean_all.append(mean)
        beta_all.append(beta)

    beta_all = numpy.asarray(beta_all)
    mean_all = numpy.asarray(mean_all)


    s_by_s_presence_absence = numpy.where(s_by_s > 0, 1, 0)
    occupancies = s_by_s_presence_absence.sum(axis=1) / s_by_s_presence_absence.shape[1]

    # calcualte total reads if no argument is passed
    # sloppy quick fix
    if len(totreads) == 0:
        totreads = s_by_s.sum(axis=0)

    # calculate mean and variance excluding zeros
    # tf = mean relative abundances
    tf = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]
        tf.append(numpy.mean(afd_no_zeros/ totreads[afd>0]))

    tf = numpy.asarray(tf)
    # go through and calculate the variance for each species

    tvpf_list = []
    for afd in s_by_s:
        afd_no_zeros = afd[afd>0]

        N_reads = s_by_s.sum(axis=0)[numpy.nonzero(afd)[0]]
        tvpf_list.append(numpy.mean(  (afd_no_zeros**2 - afd_no_zeros) / (totreads[afd>0]**2) ))

    tvpf = numpy.asarray(tvpf_list)

    f = occupancies*tf
    vf= occupancies*tvpf

    # there's this command in Jacopo's code %>% mutate(vf = vf - f^2 )%>%
    # It's applied after f and vf are calculated, so I think I can use it
    # This should be equivalent to the mean and variance including zero
    vf = vf - (f**2)

    beta = (f**2)/vf
    theta = f/beta

    richness_observed = s_by_s_presence_absence.sum(axis=0)
    richness_predicted = numpy.asarray([sum(1-((1+theta*totreads_i)**(-1*beta))) for  totreads_i in totreads])

    to_keep = ((~numpy.isnan(richness_observed)) & (~numpy.isnan(richness_predicted)))

    #richness_observed = richness_observed[to_keep]
    #richness_predicted = richness_predicted[to_keep]

    mean_richness_observed = numpy.mean(richness_observed[to_keep])
    mean_richness_predicted = numpy.mean(richness_predicted[to_keep])


    return mean_richness_observed, mean_richness_predicted



def calculate_autocorrelation(array, time, min_n_obs=10):

    # makes sure correlations are calcuated for all comparisons with the same time difference 

    rho_all = []
    delta_t_all = []

    autocorr_dict = {}
    for t in range(1, len(array)-min_n_obs+1):
        
        array_t = array[t:]
        array_delta_t = array[:-t]
        delta_t_array = time[t:] - time[:-t]

        for array_t_k_idx, array_t_k in enumerate(array_t):

            if delta_t_array[array_t_k_idx] not in autocorr_dict:
                autocorr_dict[delta_t_array[array_t_k_idx]] = {}
                autocorr_dict[delta_t_array[array_t_k_idx]]['array_t'] = []
                autocorr_dict[delta_t_array[array_t_k_idx]]['array_delta_t'] = []

            autocorr_dict[delta_t_array[array_t_k_idx]]['array_t'].append(array_t_k)
            autocorr_dict[delta_t_array[array_t_k_idx]]['array_delta_t'].append(array_delta_t[array_t_k_idx])

    delta_t_keys = list(autocorr_dict.keys())
    delta_t_keys.sort()
    for delta_t_i in delta_t_keys:

        if len(autocorr_dict[delta_t_i]['array_t']) < min_n_obs:
            continue

        rho_i = numpy.corrcoef(autocorr_dict[delta_t_i]['array_t'], autocorr_dict[delta_t_i]['array_delta_t'])[0,1]
        delta_t_all.append(delta_t_i)
        rho_all.append(rho_i)

    
    rho_all = numpy.asarray(rho_all)
    delta_t_all = numpy.asarray(delta_t_all)

       
    return rho_all, delta_t_all




#make_rrna_copy_dict()

#load_count_data()

#build_taxonomy_dict()

#build_metadata_dict()