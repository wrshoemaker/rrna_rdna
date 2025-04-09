import config
import sys
import argparse
import copy
import numpy
import utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from statsmodels.stats.multitest import fdrcorrection

from scipy import stats, signal
# numdifftools also installed
import pickle

import sine_parameter_utils



minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()
param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
gam_coeff_dict = utils.build_gam_coeff_dict()
env_dict = sine_parameter_utils.load_param_env_dict()

def plot_doc_vs_afd():

    legend_elements = [Line2D([0], [0], color=utils.dna_rna_color_dict['DNA'], label='Autotroph OTU', lw=2), 
                    Line2D([0], [0], color=utils.dna_rna_color_dict['RNA'], alpha=0.7, label='Heterotroph OTUs', lw=1), 
                    Line2D([0], [0], color='k',  label='DOC', lw=2)]

    #Line2D([0], [0], color=utils.dna_rna_color_dict['RNA'], alpha=1, label='Coarse-grained heterotroph', lw=2), 

    #param_env_dict = sine_parameter_utils.load_param_env_dict()
    s_by_s, otu_labels, samples = utils.load_count_data()
    metadata_dict = utils.build_metadata_dict()
    sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
    days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

    env_variable_array = numpy.asarray([metadata_dict[s]['water_temp'] for s in samples[(sample_type=='RNA')]])
    # remove nans
    env_to_keep_idx = (~numpy.isnan(env_variable_array))
    env_variable_array_clean = env_variable_array[env_to_keep_idx]
    days_clean = days[env_to_keep_idx]


    otu_1_idx = param_dict['otu_labels'].index('Otu000001')

    afd_days = param_dict['data']['days']['RNA'][0]
    #afd_rna_1 = param_dict['data']['clr_afd']['RNA'][otu_1_idx]
    afd_dna_1 = param_dict['data']['clr_afd']['DNA'][otu_1_idx]
    afd_rna_1 = param_dict['data']['clr_afd']['RNA'][otu_1_idx]

    #afd_ratio_1 = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_1_idx]) - numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_1_idx])

    #print(param_dict['data']['clr_afd'].keys())

    amp_1 = float(param_dict['amp_mle']['DNA'][otu_1_idx])
    timescale_1 = 2*numpy.pi/param_dict['freq_mle']['DNA'][otu_1_idx]

    #print(timescale_1*0.05)

    fig, ax = plt.subplots(figsize=(6,4))
    # ax.scatter(afd_days, afd_dna_1, s=8, alpha=1, c=utils.dna_rna_color_dict['DNA'], zorder=2)
    #ax.scatter(afd_days, afd_rna_1, s=8, alpha=1, c=utils.dna_rna_color_dict['RNA'], zorder=2)
    ax.plot(afd_days, afd_rna_1, lw=1.5, alpha=1, c=utils.dna_rna_color_dict['RNA'], zorder=2)

    ax_env = ax.twinx()
    #ax_env.scatter(days_clean, env_variable_array_clean, s=8, alpha=1, c='k', zorder=2)
    ax_env.plot(days_clean, env_variable_array_clean, lw=1.5, alpha=1, c='k', zorder=2)

    afd_all = []
    afd_coarse = numpy.zeros(len(afd_days))
    amp_sum = 0
    for otu_label_i in param_dict['otu_labels']:

        # skip phototroph
        if otu_label_i == 'Otu000001':
            #print(otu_label_i, float(param_dict['amp_mle']['DNA'][param_dict['otu_labels'].index(otu_label_i)]))
            continue

        otu_i_idx = param_dict['otu_labels'].index(otu_label_i)


        timescale_i = 2*numpy.pi/param_dict['freq_mle']['DNA'][otu_i_idx]
        amp_i = float(param_dict['amp_mle']['DNA'][otu_i_idx])
        phase_i = float(param_dict['amp_mle']['DNA'][otu_i_idx])

        #phase_rel_error_i = numpy.absolute(phase_1 - phase_i)/phase_1
        timescale_real_error_i = numpy.absolute(timescale_1 - timescale_i)/timescale_1

        #if (timescale_i < 330) or (timescale_i > 390):
        #    continue 

        if timescale_real_error_i > 0.05:
            continue

        p_value_fdr = gam_coeff_dict[otu_label_i]['dna']['doc']['p_value_fdr']

        #if p_value_fdr > 0.05:
        #    continue

        #if phase_rel_error_i > 0.1:
        #    continue

        print(otu_label_i, amp_i)

        #if amp_i < 0.8:
        #    continue

        afd_i = param_dict['data']['clr_afd']['DNA'][otu_i_idx]
        afd_all.append(afd_i)

        afd_coarse += afd_i


        # get significant p-values
        #sig_env_vars = [key for key,value in gam_coeff_dict[otu_label_i]['dna'].items() if value['p_value'] <=0.05]
        #print(otu_label_i, sig_env_vars)

        #afd_dna_i = param_dict['data']['clr_afd']['DNA'][otu_i_idx]


        #afd_ratio_i = numpy.asarray(param_dict['data']['clr_afd']['RNA'][otu_i_idx]) - numpy.asarray(param_dict['data']['clr_afd']['DNA'][otu_i_idx])
        #afd_ratio_coarse += afd_ratio_i

        #ax.plot(afd_days, afd_i, lw=0.3, alpha=0.7, c=utils.dna_rna_color_dict['RNA'], zorder=2)


    #afd_all = numpy.asarray(afd_all)
    #coarse_afg = numpy.sum(afd_all, axis=0)

    #ax.plot(afd_days, afd_ratio_coarse, lw=1, alpha=1, c=utils.dna_rna_color_dict['RNA'], zorder=2)

    #coarse_ratio_

    #ax.plot(afd_days, afd_coarse, lw=1, alpha=1, c=utils.dna_rna_color_dict['RNA'], zorder=2)
    #ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax.set_xlabel('Time (days)', fontsize=13, fontweight='bold')
    ax.set_xlim([0, max(afd_days)])
    ax.set_xticks(minor_days, minor=True)
    ax.set_xticks(major_days, minor=False)
    ax.set_xticklabels(major_labels, minor=False, fontsize=7)

    ax.set_ylabel("CLR-transformed abundance, RNA", fontsize=13, c=utils.dna_rna_color_dict['RNA'], fontweight='bold')
    ax_env.set_ylabel(utils.env_variable_label_dict['water_temp'], fontsize=13, fontweight='bold')


    fig.subplots_adjust(hspace=0.35, wspace=0.40)
    fig.savefig("%stemp_vs_afd_ratio.png" % (config.analysis_directory), format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def compare_coarse_amplitudes():

    timescale_1 = 2*numpy.pi/param_dict['freq_mle']['DNA'][param_dict['otu_labels'].index('Otu000001')]
    phase_1 = 2*numpy.pi/param_dict['phase_mle']['DNA'][param_dict['otu_labels'].index('Otu000001')]
    amp_1 = 2*numpy.pi/param_dict['amp_mle']['DNA'][param_dict['otu_labels'].index('Otu000001')]

    target_phase_all = []
    target_amp_all = []
    target_otu_label_all = []
    for otu_label_i in param_dict['otu_labels']:

        # skip phototroph
        if otu_label_i == 'Otu000001':
            continue

        otu_i_idx = param_dict['otu_labels'].index(otu_label_i)

        timescale_i = 2*numpy.pi/param_dict['freq_mle']['DNA'][otu_i_idx]
        phase_i = float(param_dict['amp_mle']['DNA'][otu_i_idx])
        amp_i = float(param_dict['amp_mle']['DNA'][otu_i_idx])

        #phase_rel_error_i = numpy.absolute(phase_1 - phase_i)/phase_1
        timescale_real_error_i = numpy.absolute(timescale_1 - timescale_i)/timescale_1

        if timescale_real_error_i > 0.05:
            continue

        p_value_fdr = gam_coeff_dict[otu_label_i]['dna']['doc']['p_value_fdr']

        if p_value_fdr > 0.05:
            continue


        target_phase_all.append(phase_i)
        target_amp_all.append(amp_i)
        target_otu_label_all.append(otu_label_i)


    env_all = ['water_temp', 'total_nitrogen', 'secchi_depth']

    doc_idx = env_dict['env_variables_labels'].index('doc')
    phase_doc = env_dict['phase_leastsq'][doc_idx]
    amp_doc = env_dict['amp_leastsq'][doc_idx]

    #print(numpy.absolute(phase_1- amp_doc) % numpy.pi)

    print(phase_1, amp_doc)

    
    print(phase_1)
    print(target_otu_label_all)
    print(target_phase_all)
    print(numpy.mean(target_phase_all))
    for env in env_all:

        env_idx = env_dict['env_variables_labels'].index(env)
        phase_env = env_dict['phase_leastsq'][env_idx]
        amp_env = env_dict['amp_leastsq'][env_idx]

        #predicted_amp = numpy.sqrt((amp_env**2) + (amp_doc**2) + 2*amp_env*amp_doc*numpy.cos(phase_doc-phase_env))
        predicted_phase = numpy.arctan((amp_1*numpy.sin(phase_1) + amp_env*numpy.sin(phase_env)) / (amp_1*numpy.cos(phase_1) + amp_env*numpy.cos(phase_env)))

        
        # we inferred the phase so that it must be between 0 and 2pi.

        predicted_phase = numpy.absolute(predicted_phase) % numpy.pi
        #print(env, predicted_phase)
    
        #print(amp_1)





def plot_():



#plot_doc_vs_afd()
