
import config 
from datetime import datetime
import numpy
import re
#import pandas
import math
import sympy
from scipy.stats import gmean
from scipy import stats


from matplotlib import cm
from collections import Counter

from statsmodels.stats.multitest import fdrcorrection

numpy.random.seed(123456789)


gam_path = '%sgam_env_analysis_only_time.csv' % config.data_directory
sub_plot_labels = ['a','b','c', 'd','e','f', 'g','h','i', 'j','k','l', 'm','n','o', 'p','q','r']


# colors_dict = {'0':'#87CEEB', '1': '#FFA500', '2':'#FF6347'}

dna_rna_color_dict = {'RNA': '#FF6347', 'DNA': '#87CEEB', 'ratio':'k', 'RNA_DNA':'k'}

color_radius = 2

cmap_data_type_dict = {'DNA': 'Blues', 'RNA': 'Reds'}
transformation_color_dict = {'rel': '#FFA500', 'clr': '#1f7e3b'}
# 13d14c

# ASV_001, ASV_006
phototroph_asv_all = ['TACGGAGGATGCAAGCGTTATCCGGAATGATTGGGCGTAAAGGGTCCGCAGGTGGCATTGTAAGTCTGCTGTTAAAGAGTTTGGCTCAACCAAATAAGAGCAGTGGAAACTACAAAGCTAGAGTGTGGTCGGGGCAGAGGGAATTCCTGGTGTAGCGGTGAAATGCGTAGATATCAGGAAGAACACCAGTGGCGAAGGCGCTCTGCTAGGCCGAGACTGACACTGAGGGACGAAAGCTAGGGGAGCGAATGGG', 'TACGGGGGATGCAAGCGTTATCCGGAATGATTGGGCGTAAAGAGTCCGTAGGTAGTCATCCAAGTCTGCTGTTAAAGAGCGAGGCTTAACCTCGTAAAGGCAGTGGAAACTGGAAGACTAGAGTGTAGTAGGGGCAGAGGGAATTCCTGGTGTAGCGGTGAAATGCGTAGAGATCAGGAAGAACACCGGTGGCGAAGGCGCTCTGCTGGGCTATAACTGACACTGAGGGACGAAAGCTAGGGGAGCGAATGGG']

rescaled_label_dict = {'RNA':'Rescaled rRNA, ' + r'$r_{i}(t)$', 'DNA': 'Rescaled rDNA, ' + r'$d_{i}(i)$', 'ratio': 'Rescaled rRNA:rDNA, ' + r'$\phi_{i}(t)$'}
#rescaled_label_dict = {'RNA':'Rescaled RNA, ' + r'$r_{i}(t)$', 'DNA': 'Rescaled DNA, ' + r'$d_{i}(i)$', 'ratio': 'Rescaled RNA:DNA, ' + r'$\phi_{i}(t)$'}
rescaled_label_clr_dict = {'RNA':'rRNA', 'DNA': 'rDNA', 'ratio': 'rRNA - rDNA'}

sample_label_dict = {'RNA': 'rRNA', 'DNA':'rDNA', 'ratio': 'rRNA:rDNA', 'RNA_DNA': 'rRNA:DNAr'}

data_type_all = ['DNA', 'RNA', 'ratio']
env_variables_all = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph']


#taxonomic_ranks = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
taxonomic_ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']



env_variable_all = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph', 'air_temperature', 'total_nitrogen', 'total_phosphorus', 'doc']
env_variable_to_plot = ['water_temp', 'specific_conductivity', 'dissolved_oxygen', 'salinity', 'secchi_depth', 'ph', 'total_nitrogen', 'total_phosphorus', 'doc']


env_variable_label_dict = {'water_temp': 'Water temperature (°C)', 'specific_conductivity': 'Specific conductivity (mS/cm)', 
                            'dissolved_oxygen': 'Dissolved oxygen (mg/L)', 'salinity': 'Salinity (PSS)',
                            'secchi_depth': 'Secchi depth (m)', 'ph':'pH', 'air_temperature': 'Air temperature (°C)', 'total_nitrogen':'Total nitrogen (mg/L)', 'total_phosphorus':'Total phosphorus (' + r'$\mu$' + 'g/L)', 'doc': 'Dissolved organic carbon (mg/L)'}


env_variable_no_unit_label_dict = {'water_temp': 'Water temperature', 'specific_conductivity': 'Specific conductivity', 
                            'dissolved_oxygen': 'Dissolved oxygen', 'salinity': 'Salinity',
                            'secchi_depth': 'Secchi depth', 'ph':'pH', 'air_temperature': 'Air temperature',  'total_nitrogen':'Total nitrogen', 'total_phosphorus':'Total phosphorus', 'doc': 'Dissolved organic carbon'}


env_variable_no_unit_label_abbrev_dict = {'water_temp': 'Water temp.', 'specific_conductivity': 'Specific cond.', 
                            'dissolved_oxygen': 'Dissolved ' + r'O$_2$', 'salinity': 'Salinity',
                            'secchi_depth': 'Secchi depth', 'ph':'pH', 'air_temperature': 'Air temp.',  'total_nitrogen':'Total N', 'total_phosphorus':'Total P', 'doc': 'DOC'}




env_variable_no_unit_label_split_dict = {'water_temp': 'Water\ntemp.', 'specific_conductivity': 'Specific\nconductivity', 
                            'dissolved_oxygen': 'Dissolved\noxygen', 'salinity': 'Salinity',
                            'secchi_depth': 'Secchi\ndepth', 'ph':'pH', 'air_temperature': 'Air\ntemperature',  'total_nitrogen':'Total N', 'total_phosphorus':'Total C', 'doc': 'Dissolved\norganic C'}


family_trophic_status = {
    "Nostocaceae": "phototroph",        # Cyanobacteria; oxygenic photosynthesis
    "Sporichthyaceae": "heterotroph",   # Actinobacteria; aerobic organotrophs
    "Comamonadaceae": "heterotroph",    # Betaproteobacteria; aerobic organotrophs
    "Phormidiaceae": "phototroph",      # Cyanobacteria; oxygenic photosynthesis
    "Chitinophagaceae": "heterotroph",  # Bacteroidota; aerobic organotrophs
    "Saprospiraceae": "heterotroph",    # Bacteroidota; aerobic organotrophs
    "Methylophilaceae": "heterotroph",  # Betaproteobacteria; methylotrophs (C1 compounds)
    "Spirosomataceae": "heterotroph",   # Bacteroidota; aerobic organotrophs
    "Alcaligenaceae": "heterotroph",    # Betaproteobacteria; aerobic organotrophs
    "Burkholderiaceae": "heterotroph",  # Betaproteobacteria; aerobic organotrophs
}


#otu_producer_status = {'Otu000001': 'phototroph': ''}



def empirical_survival(sample):
    x = numpy.sort(numpy.asarray(sample))
    n = len(x)
    S = 1 - numpy.arange(1, n + 1) / n
    return x, S


def get_p_value_latex_label_dict(p_value):

    if p_value <= 0.05:
        label = r'$P \, \leq \, 0.05$'


    else:
        label = r'$P \, \nleq \, 0.05$'

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




def parse_time(line):

    # Extract time of day and hours since midnight 
    # Returns (time_str, hours_since_midnight) or (None, None)

    # Normalize semicolons used as colons (e.g. "19;19").......
    line_norm = re.sub(r'(\d{1,2});(\d{2})', r'\1:\2', line)

    # Try 12-hour format first: 10:00 AM, 2:45 PM
    match = re.search(r'\b(\d{1,2}):(\d{2})\s*(AM|PM)\b', line_norm, re.IGNORECASE)
    if match:
        h, m, meridiem = int(match.group(1)), int(match.group(2)), match.group(3).upper()
        if meridiem == 'PM' and h != 12:
            h += 12
        elif meridiem == 'AM' and h == 12:
            h = 0
        hours = h + m / 60
        return f"{match.group(0)}", hours

    # Try 24-hour format: 13:22, 08:26, 15:00
    match = re.search(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', line_norm)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        hours = h + m / 60

        if hours == 1.75:
            hours = 11.75
        
        return f"{match.group(0)}", hours


    
    return None, None



def build_metadata_dict(return_srr_dict=False):

    # change for ASVs
    #file_ = open('%sdesign.csv' % config.data_directory, 'r')
    file_ = open('%ssra_metadata_annotated.csv' % config.data_directory, 'r')
    file_header = file_.readline()
    
    metadata_dict = {}

    sample_meta_formate_to_srr = {}
    for line in file_:

        line = line.strip().split(',')

        sample_meta = line[11].strip('"')
        sample = line[0].strip('"')
        #sample_type = line[1].strip('"')
        sample_type = line[-1].strip('"')

        metadata_dict[sample] = {}
        metadata_dict[sample]['sample_type'] = sample_type
        metadata_dict[sample]['sample_meta'] = sample_meta
        sample_meta_formate_to_srr[sample_meta] = sample

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
            days = (datetime_object - metadata_dict[sample_meta_formate_to_srr['ULc001']]['date']).days

        time_of_day_str = line[-1]
        # non-standardized format
        time_str, hours_since_midnight = parse_time(time_of_day_str)

        if hours_since_midnight is not None:
            hours_since_midnight = float(hours_since_midnight)

        rna_sample = 'ULc' + sample
        dna_sample = 'ULD' + sample

        for sample_i in [rna_sample, dna_sample]:

            sample_srr_i = sample_meta_formate_to_srr[sample_i]

            metadata_dict[sample_srr_i]['date'] = datetime_object
            metadata_dict[sample_srr_i]['day_of_year'] = datetime_object.timetuple().tm_yday
            metadata_dict[sample_srr_i]['day'] = days
            metadata_dict[sample_srr_i]['depth'] = depth
            metadata_dict[sample_srr_i]['water_temp'] = water_temp
            metadata_dict[sample_srr_i]['specific_conductivity'] = specific_conductivity
            metadata_dict[sample_srr_i]['dissolved_oxygen'] = dissolved_oxygen
            metadata_dict[sample_srr_i]['salinity'] = salinity
            metadata_dict[sample_srr_i]['secchi_depth'] = secchi_depth
            metadata_dict[sample_srr_i]['ph'] = ph
            metadata_dict[sample_srr_i]['air_temperature'] = air_temperature
            metadata_dict[sample_srr_i]['hours_since_midnight'] = hours_since_midnight

    
    file_environment.close()

    # get nitrogen, phosphorous, and DOC
    file_environment_2 = open('%sul-seedbank.env.txt' % config.data_directory, 'r')
    file_environment_2_header = file_environment_2.readline()
    file_environment_2_header = file_environment_2_header.strip().split('\t')
    
    for line in file_environment_2:

        line = line.strip().split('\t')
        sample = line[0]

        if len(sample) == 1:
            sample = '00' + sample

        elif len(sample) == 2:
            sample = '0' + sample

        else:
            sample = sample


        doc = line[-3]
        total_nitrogen = line[-4]
        total_phosphorus = line[-5]

        if (len(doc) > 0) and ('*' not in doc):
            doc = float(doc)
        else:
            doc = numpy.nan

        if (len(total_nitrogen) > 0) and ('*' not in total_nitrogen):
            total_nitrogen = float(total_nitrogen)
        else:
            total_nitrogen = numpy.nan

        if (len(total_phosphorus) > 0) and ('*' not in total_phosphorus):
            total_phosphorus = float(total_phosphorus)
        else:
            total_phosphorus = numpy.nan


        # ignore low numbers of outliers

        if total_nitrogen > 3:
            total_nitrogen = numpy.nan

        if total_phosphorus > 140:
            total_phosphorus = numpy.nan

        if (doc > 8) or (doc < 1):
            doc = numpy.nan


        rna_sample = 'ULc' + sample
        dna_sample = 'ULD' + sample

        if dna_sample not in sample_meta_formate_to_srr:
            continue

        sample_srr_rna_i = sample_meta_formate_to_srr[rna_sample]
        sample_srr_dna_i = sample_meta_formate_to_srr[dna_sample]

        #if sample_srr_dna_i not in metadata_dict:
        #    continue
        
        for sample_i in [rna_sample, dna_sample]:
            
            sample_srr_i = sample_meta_formate_to_srr[sample_i]

            metadata_dict[sample_srr_i]['doc'] = doc
            metadata_dict[sample_srr_i]['total_nitrogen'] = total_nitrogen
            metadata_dict[sample_srr_i]['total_phosphorus'] = total_phosphorus


    file_environment_2.close()

    if return_srr_dict == True:
        return metadata_dict, sample_meta_formate_to_srr
    
    else:
        return metadata_dict


def build_taxonomy_dict_otu():

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




def build_taxonomy_dict_indiv(filepath):
    taxonomy_dict = {}

    with open(filepath, 'r') as f:
        # skip header
        header = f.readline()  

        for line in f:
            parts = line.strip().split('\t')
            asv  = parts[0]
            taxa = parts[1:]

            taxonomy_dict[asv] = {rank: (t if t != '' else 'NA') for rank, t in zip(taxonomic_ranks, taxa)}

    return taxonomy_dict


def build_taxonomy_dict():
    
    taxonomy_dict = build_taxonomy_dict_indiv('%sdada2/DNA/taxa_DNA.txt' % config.data_directory)
    taxonomy_dict.update(build_taxonomy_dict_indiv('%sdada2/RNA/taxa_RNA.txt' % config.data_directory))

    return taxonomy_dict


def load_count_data_otu():

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
    

def load_seqtab(filepath):
    
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()

    header = lines[0].split('\t')
    # Header may or may not have a leading empty field before sample names
    sample_names = header[1:] if header[0] == '' else header

    asv_names = []
    counts = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split('\t')
        asv_names.append(parts[0])
        counts.append([int(x) for x in parts[1:]])

    count_matrix = numpy.array(counts, dtype=numpy.int64)

    return count_matrix, asv_names, sample_names


def load_count_data():

    mat1, asvs1, samples1 = load_seqtab('%sdada2/DNA/seqtab_nochim_DNA.txt' % config.data_directory)
    mat2, asvs2, samples2 = load_seqtab('%sdada2/RNA/seqtab_nochim_RNA.txt' % config.data_directory)

    # ordered union
    asv_names = list(dict.fromkeys(asvs1 + asvs2))
    asv_index = {asv: i for i, asv in enumerate(asv_names)}

    n_asvs    = len(asv_names)
    n_samples = len(samples1) + len(samples2)
    s_by_s = numpy.zeros((n_asvs, n_samples), dtype=numpy.int64)

    for local_i, asv in enumerate(asvs1):
        global_i = asv_index[asv]
        s_by_s[global_i, :len(samples1)] = mat1[local_i]

    #  offset columns by number of samples in file 1
    col_offset = len(samples1)
    for local_i, asv in enumerate(asvs2):
        global_i = asv_index[asv]
        s_by_s[global_i, col_offset:] = mat2[local_i]

    sample_names = samples1 + samples2

    asv_names = numpy.array(asv_names)
    sample_names = numpy.asarray(sample_names)


    return s_by_s, asv_names, sample_names




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






def predict_occupancy_provide_mean(s_by_s_mean, s_by_s_occupancy):

    # get squared inverse cv
    # assume that entries are read counts.

    totreads_mean = s_by_s_mean.sum(axis=0)
    totreads_occupancy = s_by_s_occupancy.sum(axis=0)

    occupancies_1 = numpy.where(s_by_s_mean > 0, 1, 0).sum(axis=1) / s_by_s_mean.shape[1]
    occupancies_2 = numpy.where(s_by_s_occupancy > 0, 1, 0).sum(axis=1) / s_by_s_occupancy.shape[1]


    # calculate mean and variance excluding zeros
    # tf = mean relative abundances
    tf = []
    for afd in s_by_s_mean:
        afd_no_zeros = afd[afd>0]
        tf.append(numpy.mean(afd_no_zeros/ totreads_mean[afd>0]))

    tf = numpy.asarray(tf)
    # go through and calculate the variance for each species

    tvpf_list = []
    for afd in s_by_s_mean:
        afd_no_zeros = afd[afd>0]
        tvpf_list.append(numpy.mean(  (afd_no_zeros**2 - afd_no_zeros) / (totreads_mean[afd>0]**2) ))

    tvpf = numpy.asarray(tvpf_list)

    f = occupancies_1*tf
    vf= occupancies_1*tvpf

    vf = vf - (f**2)
    beta = (f**2)/vf
    theta = f/beta

    #predicted_occupancies_diff = []
    predicted_occupancies_1 = []
    predicted_occupancies_2 = []
    # each species has it's own beta and theta, which is used to calculate predicted occupancy
    for beta_i, theta_i in zip(beta,theta):
        predicted_occupancies_1.append(1 - numpy.mean( ((1+theta_i*totreads_mean)**(-1*beta_i ))   ))
        predicted_occupancies_2.append(1 - numpy.mean( ((1+theta_i*totreads_occupancy)**(-1*beta_i ))   ))

    predicted_occupancies_1 = numpy.asarray(predicted_occupancies_1)
    predicted_occupancies_2 = numpy.asarray(predicted_occupancies_2)

    rel_s_by_s = (s_by_s_mean/s_by_s_mean.sum(axis=0))
    mad = numpy.mean(rel_s_by_s, axis=1)
    
    return occupancies_1, occupancies_2, predicted_occupancies_1, predicted_occupancies_2, mad, beta





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



def clr_transform_subset(s_by_s, otu_labels, samples, min_occupancy=1):

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

    # relative abundance within each sample
    rel_rna = s_by_s_rna_occupancy / numpy.sum(s_by_s_rna_occupancy, axis=0)
    rel_dna = s_by_s_dna_occupancy / numpy.sum(s_by_s_dna_occupancy, axis=0)

    # concatenate all samples
    rel_all = numpy.concatenate([rel_rna, rel_dna], axis=1)

    # mean relative abundance across time + RNA/DNA
    mean_rel_abundance = numpy.mean(rel_all, axis=1)

    # descending sort
    sort_idx = numpy.argsort(mean_rel_abundance)[::-1]

    s_by_s_rna_occupancy = s_by_s_rna_occupancy[sort_idx, :]
    s_by_s_dna_occupancy = s_by_s_dna_occupancy[sort_idx, :]
    otu_labels_occupancy = otu_labels_occupancy[sort_idx]

    #n_reads_rna = numpy.sum(s_by_s_rna, axis=0)
    #n_reads_dna = numpy.sum(s_by_s_dna, axis=0)

    n_reads_rna_gmean = gmean(s_by_s_rna_occupancy, axis=0)
    n_reads_dna_gmean = gmean(s_by_s_dna_occupancy, axis=0)

    clr_s_by_s_rna = numpy.log(s_by_s_rna_occupancy/n_reads_rna_gmean)
    clr_s_by_s_dna = numpy.log(s_by_s_dna_occupancy/n_reads_dna_gmean)

    return clr_s_by_s_dna, clr_s_by_s_rna, otu_labels_occupancy



def clr_transform(s_by_s, otu_labels, samples, min_occupancy = 1, pseudocount = 1):

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

    #s_by_s_rna_pseud = s_by_s_rna + pseudocount
    #s_by_s_dna_pseud = s_by_s_dna + pseudocount
    
    s_by_s_rna_occupancy = s_by_s_rna[occupancy_idx,:]
    s_by_s_dna_occupancy = s_by_s_dna[occupancy_idx,:]


    n_reads_dna_occupancy = numpy.sum(s_by_s_dna_occupancy, axis=0)
    n_reads_rna_occupancy = numpy.sum(s_by_s_rna_occupancy, axis=0)


    # geometric mean *over OTUs* per-sample
    # length of vector is # of samples
    n_reads_rna_gmean = gmean(s_by_s_rna_occupancy, axis=0)
    n_reads_dna_gmean = gmean(s_by_s_dna_occupancy, axis=0)

    clr_s_by_s_rna = (numpy.log(s_by_s_rna_occupancy) - numpy.log(n_reads_rna_gmean))
    clr_s_by_s_dna = (numpy.log(s_by_s_dna_occupancy) - numpy.log(n_reads_dna_gmean))

    # return 

    return clr_s_by_s_dna, clr_s_by_s_rna, occupancy_idx, otu_labels_occupancy, n_reads_dna_occupancy, n_reads_rna_occupancy



def clr_transform_sim_subset(s_by_s, min_occupancy=1):

    # requires rel_s_by_s ASVs to all have occupancy = 1
    occupancy = numpy.sum((s_by_s>0), axis=1)/s_by_s.shape[1]

    occupancy_idx = (occupancy>=min_occupancy) #& (occupancy>=min_occupancy)
    
    s_by_s_occupancy = s_by_s[occupancy_idx,:]

    # geometric mean *across* species calculated for each sample
    # len(n_reads_gmean) = # samples
    n_reads_gmean = gmean(s_by_s_occupancy, axis=0)

    clr_s_by_s = numpy.log(s_by_s_occupancy/n_reads_gmean)

    return clr_s_by_s, occupancy_idx



def clr_transform_sim(s_by_s, pseudocount=1, min_occupancy=1):

    # requires rel_s_by_s ASVs to all have occupancy = 1
    occupancy = numpy.sum((s_by_s>0), axis=1)/s_by_s.shape[1]

    occupancy_idx = (occupancy>=min_occupancy) #& (occupancy>=min_occupancy)
    
    s_by_s_pseud = s_by_s + pseudocount
    
    # geometric mean *over OTUs* per-sample
    n_reads_gmean = gmean(s_by_s_pseud, axis=0)

    clr_s_by_s = (numpy.log(s_by_s_pseud) - numpy.log(n_reads_gmean))

    # geometric mean *across* species calculated for each sample

    return clr_s_by_s, occupancy_idx




def get_seasonal_tick_labels():

    # returns minor ticks, major ticks , and major tick labels

    # x axis in unit of days
    metadata_dict = build_metadata_dict()

    dna_samples = [k for k in metadata_dict.keys() if 'DNA' in metadata_dict[k]['sample_type']]
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
    n_all = []

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

        n_all.append(len(autocorr_dict[delta_t_i]['array_t']))

    
    rho_all = numpy.asarray(rho_all)
    delta_t_all = numpy.asarray(delta_t_all)
    n_all = numpy.asarray(n_all)

       
    return rho_all, delta_t_all, n_all



def corr_permute_test(x, y, n_iter=10000):

    x = numpy.asarray(x)
    y = numpy.asarray(y)

    rho_obs = (numpy.corrcoef(x, y)[0,1])**2

    rho_null_all = []

    for i in range(n_iter):

        x_null = numpy.random.permutation(x)
        y_null = numpy.random.permutation(y)

        rho_null_all.append((numpy.corrcoef(x_null, y_null)[0,1])**2)

    rho_null_all = numpy.asarray(rho_null_all)

    p_value = sum(rho_null_all > rho_obs)/n_iter

    return rho_obs, p_value




def build_gam_coeff_dict(floor_p_value=1e-8):

    gam_coeff_dict = {}

    gam_env_analysis_path = '%sgam_env_analysis.csv' % config.data_directory

    gam_env_analysis_file = open(gam_env_analysis_path, 'r')
    header = gam_env_analysis_file.readline()
    env_variables = header.strip().split(',')[2:]

    for line in gam_env_analysis_file:

        line = line.strip().split(',')

        line_0_split = line[0].split('_')
        #otu = line[0].split('_', 1)[1]
        #otu = line_0_split[1]
        #data_type = line[0].split('_', 1)[-1]
        #print(len(line_0_split))
        #data_type = line_0_split[-1]
        
        otu = line_0_split[1]
        for suffix in ['rna_dna', 'rna', 'dna']:
            if line[0].endswith('_' + suffix):
                data_type = suffix
                break


        if otu not in gam_coeff_dict:
            gam_coeff_dict[otu] = {}

            for d in ['dna', 'rna', 'rna_dna']:
                gam_coeff_dict[otu][d] = {}

                for e in env_variables:
                    gam_coeff_dict[otu][d][e] = {}

        p_value_or_coeff = line[1]
        for e_idx, e in enumerate(env_variables):

            
            gam_coeff_dict[otu][data_type][e][p_value_or_coeff] = float(line[e_idx+2])

    gam_env_analysis_file.close()
    
    otu_list = list(gam_coeff_dict.keys())

    for data_type in ['rna', 'dna', 'rna_dna']:

        for e_idx, e in enumerate(env_variables):

            p_value = numpy.asarray([gam_coeff_dict[k][data_type][e]['p_value'] for k in otu_list])
            p_value_fdr = fdrcorrection(p_value, alpha=0.05, method='indep', is_sorted=False)[1]
            for k_idx, k in enumerate(otu_list):
                gam_coeff_dict[k][data_type][e]['p_value_fdr'] = p_value_fdr[k_idx]


    return gam_coeff_dict
    


# performs regression on the two arrays passes
def get_confidence_hull(x, y, conf=0.95):

    if type(x) is not numpy.ndarray:
        x = numpy.asarray(x)

    if type(y) is not numpy.ndarray:
        y = numpy.asarray(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)


    if min([min(x), min(y) ] ) < 0:
        min_range = min([min(x), min(y) ] ) * 1.3

    else:
        min_range = min([min(x), min(y) ] ) * 0.5

    max_range = max([max(x), max(y) ] ) * 1.3

    x_range = numpy.linspace(min_range, max_range, num=1000)
    y_range_pred = numpy.asarray([ intercept + (x_i*slope) for x_i in  x_range])

    y_pred = numpy.asarray([intercept + (slope*x_i) for x_i in x])

    SSE = sum((y - y_pred) ** 2)
    N = len(x)
    sd_SSE = numpy.sqrt( (1/ (N-2)) * SSE)
    sxd = numpy.sum((x-numpy.mean(x))**2)

    sx = (x_range-numpy.mean(x))**2	# x axisr for band
    # Quantile of Student's t distribution for p=1-alpha/2
    alpha = 1-conf
    q = stats.t.ppf(1-alpha/2, N-2)
    # Confidence band
    dy = q*sd_SSE*numpy.sqrt( 1/N + sx/sxd )
    # Upper confidence band
    ucb = y_range_pred + dy
    # Lower confidence band
    lcb = y_range_pred - dy


    return x_range, y_range_pred, lcb, ucb




def load_gam():

    s_by_s, otu_labels, samples = load_count_data()
    metadata_dict = build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    env_var_dict = {}
    for env_variable_idx, env_variable in enumerate(env_variable_all):
        
        env_variable_array = numpy.asarray([metadata_dict[s][env_variable] for s in samples[(sample_type=='RNA')]])
        # remove nans
        env_to_keep_idx = (~numpy.isnan(env_variable_array))
        env_variable_array_clean = env_variable_array[env_to_keep_idx]
        env_var_dict[env_variable] = numpy.std(env_variable_array_clean)


    gam_dict = {}
    gam_file = open(gam_path, 'r')
    gam_header = gam_file.readline().strip().split(',')
    for line in gam_file:

        if 'Otu000001' not in line:
            continue

        line = line.strip().split(',')
        data_type = line[0].split('_', 1)[1]

        stat = line[1]

        for env_variable_idx in range(2, len(line)):

            env_variable = gam_header[env_variable_idx]

            if env_variable not in gam_dict:
                gam_dict[env_variable] = {}

            if data_type not in gam_dict[env_variable]:
                gam_dict[env_variable][data_type] = {}
                    
            gam_dict[env_variable][data_type][stat] = float(line[env_variable_idx])

            # rescale
            if stat == 'coeff':
                gam_dict[env_variable][data_type]['coeff_scaled'] = float(line[env_variable_idx]) * env_var_dict[env_variable]
            else:
                gam_dict[env_variable][data_type]['p_value'] = float(line[env_variable_idx])

    gam_file.close()

    #print(gam_dict.keys())

    # FDR correction
    env_variable_all_ = list(gam_dict.keys())
    for data_type in ['dna', 'rna', 'rna_dna']:

        p_value_all = numpy.asarray([gam_dict[e][data_type]['p_value'] for e in env_variable_all_])
        p_value_all_corrected = fdrcorrection(p_value_all, alpha=0.05, method='indep', is_sorted=False)[1]

        for env_variable_idx, env_variable in enumerate(env_variable_all_):

            gam_dict[env_variable][data_type]['p_value_fdr'] = p_value_all_corrected[env_variable_idx]



    return gam_dict



def make_asv_fasta(n_fna_characters=config.n_fna_characters, min_occupancy=0.2):

    # creates fasta file from ASV names
    s_by_s, asv_names, sample_names = load_count_data()

    occupancy = (s_by_s>0).sum(axis=1) / s_by_s.shape[1]

    asv_names = numpy.asarray(asv_names)
    asv_names_final = asv_names[occupancy >= min_occupancy]

    out_path = '%sasv.fna' % config.data_directory
    out_file = open(out_path, 'w')

    for asv_sequence in asv_names_final:

        out_file.write('>%s\n' % asv_sequence)

        for i in range(0, len(asv_sequence), n_fna_characters):
            asv_sequence_i = asv_sequence[i : i + n_fna_characters]
            out_file.write('%s\n' % asv_sequence_i)
        out_file.write('\n')

    out_file.close()





class classFASTA:

    # class to load FASTA file

    def __init__(self, fileFASTA):
        self.fileFASTA = fileFASTA

    def readFASTA(self):
        '''Checks for fasta by file extension'''
        file_lower = self.fileFASTA.lower()
        '''Check for three most common fasta file extensions'''
        if file_lower.endswith('.txt') or file_lower.endswith('.fa') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.fna') or \
        file_lower.endswith('.fasta') or file_lower.endswith('.frn') or \
        file_lower.endswith('.faa') or file_lower.endswith('.ffn'):
            with open(self.fileFASTA, "r") as f:
                return self.ParseFASTA(f)
        else:
            print("Not in FASTA format.")

    def ParseFASTA(self, fileFASTA):
        '''Gets the sequence name and sequence from a FASTA formatted file'''
        fasta_list=[]
        for line in fileFASTA:
            if line[0] == '>':
                try:
                    fasta_list.append(current_dna)
            	#pass if an error comes up
                except UnboundLocalError:
                    #print "Inproper file format."
                    pass
                current_dna = [line.lstrip('>').rstrip('\n'),'']
            else:
                current_dna[1] += "".join(line.split())
        fasta_list.append(current_dna)
        '''Returns fasa as nested list, containing line identifier \
            and sequence'''
        return fasta_list
    




def clean_alignment(muscle_path, muscle_clean_path, min_n_sites=100, max_fraction_empty=0.8):

    # removes all sites where the fraction of empty bases across ASVs is greater than max_fraction_empty (putatively uninformative)
    # removes a sequencies with fewer than min_n_sites informative sites
    frn_aligned = classFASTA(muscle_path).readFASTA()

    n = len(frn_aligned)

    frn_aligned_seqs = [x[1] for x in frn_aligned]
    frn_aligned_seqs_names = [x[0] for x in frn_aligned]

    frns = []
    for site in zip(*frn_aligned_seqs):

        fraction_empty = site.count('-')/n

        if fraction_empty > max_fraction_empty:
            continue

        # skip site if it is uninformative
        if len(set([s for s in site if s != '-'])) == 1:
            continue

        frns.append(site)

    if len(frns) < min_n_sites:
        exit()

    clean_sites_list = zip(*frns)

    frn_aligned_clean = open(muscle_clean_path, 'w')
    for clean_sites_idx, clean_sites in enumerate(clean_sites_list):
        clean_sites_species = frn_aligned_seqs_names[clean_sites_idx]
        clean_sites_seq = "".join(clean_sites)

        frn_aligned_clean.write('>%s\n' % clean_sites_species)

        clean_sites_seq_split = [clean_sites_seq[i:i+config.n_fna_characters] for i in range(0, len(clean_sites_seq), config.n_fna_characters)]

        for seq in clean_sites_seq_split:

            frn_aligned_clean.write('%s\n' % seq)

        frn_aligned_clean.write('\n')


    frn_aligned_clean.close()





def compute_pvalue(observed, null_distribution, side="two"):

    null = numpy.asarray(null_distribution)

    p_right = numpy.mean(null >= observed)
    p_left  = numpy.mean(null <= observed)

    if side == "right":
        return p_right
    elif side == "left":
        return p_left
    elif side == "two":
        return 2 * min(p_left, p_right)
    else:
        raise ValueError("side must be 'right', 'left', or 'two'")



def perm_slope(x, y, n_perm=10000):

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # permutation test
    rng = numpy.random.default_rng(123)

    perm_slopes = numpy.empty(n_perm)

    for i in range(n_perm):
        y_perm = rng.permutation(y)
        perm_slopes[i] = stats.linregress(x, y_perm).slope

    # two-sided p-value
    p_perm = (numpy.sum(numpy.abs(perm_slopes) >= numpy.abs(slope)) + 1) / (n_perm + 1)

    return slope, intercept, r_value, p_perm, std_err




def rayleigh_test(angles):
    # Rayleigh test for circular uniformity
    # H0: angles are uniformly distributed on the circle
    # Returns: R (mean resultant length), p-value

    n = len(angles)
    # mean resultant vector
    C = numpy.mean(numpy.cos(angles))
    S = numpy.mean(numpy.sin(angles))
    R = numpy.sqrt(C**2 + S**2)  # mean resultant length, 0 to 1
    
    # Rayleigh test statistic
    z = n * R**2
    
    # p-value (approximation valid for n > 10)
    p = numpy.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    
    return R, p



if __name__ == "__main__":

    print('Utility file')

    #make_asv_fasta(min_occupancy=1)
    #muscle_path = '%sasv_w_outgroup_aligned.fna' % config.data_directory
    #muscle_clean_path = '%sasv_w_outgroup_aligned_clean.fna' % config.data_directory
    #clean_alignment(muscle_path, muscle_clean_path)

    #m_dict = build_metadata_dict()
    #samples = list(m_dict.keys())

    #time = numpy.array([m_dict[s]['hours_since_midnight'] for s in samples], dtype=float) 
    #day_of_year = numpy.array([m_dict[s]['day_of_year'] for s in samples], dtype=float) 
    #to_keep_idx = ~numpy.isnan(time)
    #time_no_nan = time[to_keep_idx]
    #day_of_year_no_nan = day_of_year[to_keep_idx]

    #print(numpy.corrcoef(day_of_year_no_nan, time_no_nan))

    #print(len(time_no_nan)/len(time))

    #print(numpy.mean(time_no_nan), numpy.std(time_no_nan))


    build_metadata_dict()