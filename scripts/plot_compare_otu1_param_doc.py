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








def plot_timescale_vs_coeff():

    sine_param_to_plot = 'freq_mle'

    fig = plt.figure(figsize = (8.5, 8)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    for env_variable_idx, env_variable in enumerate(['doc', 'ph']):

        for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

            sine_param_all = []
            coeff_all = []
            pvalue_all = []

            for otu_label_idx, otu_label in enumerate(param_dict['otu_labels']):

                if otu_label_idx == 0:
                    continue

                param_otu = gam_coeff_dict[otu_label][data_type.lower()][env_variable]['coeff']

                sine_param_all.append(param_dict[sine_param_to_plot][data_type][otu_label_idx])
                coeff_all.append(param_otu)
                pvalue_all.append(gam_coeff_dict[otu_label][data_type.lower()][env_variable]['p_value'])


            focal_otu_sine_param = param_dict[sine_param_to_plot][data_type][0]
            focal_otu_coeff = gam_coeff_dict[param_dict['otu_labels'][0]][data_type.lower()][env_variable]['coeff']

            sine_param_all = numpy.asarray(sine_param_all)
            coeff_all = numpy.asarray(coeff_all)
            pvalue_all = numpy.asarray(pvalue_all)
            
            # absolute value

            if 'freq' in sine_param_to_plot:

                sine_param_all = 2*numpy.pi/sine_param_all
                focal_otu_sine_param = 2*numpy.pi/focal_otu_sine_param


            coeff_all = numpy.absolute(coeff_all)
            focal_otu_coeff = abs(focal_otu_coeff)



            pvalue_all = fdrcorrection(pvalue_all, alpha=0.05, method='indep', is_sorted=False)[1]

            ax = fig.add_subplot(gs[env_variable_idx, data_type_idx])
            ax.scatter(focal_otu_sine_param, focal_otu_coeff, alpha=1, s=30, color='k', label='OTU 1 (phototroph)')

            pvalue_significant_idx = (pvalue_all <= 0.05)

            edgecolor = utils.dna_rna_color_dict[data_type]

            for sig_bool_ in [True, False]:

                # skip if there are no significant slopes...
                if sum(pvalue_significant_idx==sig_bool_) == 0:
                    continue

                if sig_bool_ == True:
                    sig_bool_label = r'$P<0.05$'
                    #color = '#87CEEB'
                    facecolor = utils.dna_rna_color_dict[data_type]
                    

                else:
                    #sig_bool_label = 'nonsignificant'
                    sig_bool_label = r'$P \, \nleq \, 0.05$'
                    #color = 'k'
                    facecolor = 'none'
                    


                ax.scatter(sine_param_all[pvalue_significant_idx==sig_bool_], coeff_all[pvalue_significant_idx==sig_bool_], alpha=0.8, s=20, edgecolors=edgecolor, facecolors=facecolor, label='Heterotrophic OTUs, %s' % sig_bool_label, zorder=2)



            ax.axhline(y=0, lw=2.5, ls=':', color='k', zorder=1)
            #min_x, ma


            ax.set_xlabel("Oscillation timescale (days), " + r'$\tau_{i}^{\mathrm{env}}$', fontsize=12)
            #ax.set_ylabel("GAM coefficient for %s" % env_variable, fontsize=12)
            ax.set_ylabel("Absolute value of GAM coefficient", fontsize=11)
            #ax.axvline(x=focal_otu_sine_param, lw=2.5, ls=':', label='OTU1', color='k', zorder=1)

            #ax.hlines(y=focal_otu_coeff, xmin=min(), xmax=1.0, color='b')

            if env_variable_idx == 0:
                ax.set_title(data_type, fontsize=16)

            if data_type_idx == 0:
                ax.text(-0.32, 0.5, utils.env_variable_label_dict[env_variable], fontsize=14, ha='center', va='center', rotation=90, transform=ax.transAxes)

            if env_variable_idx + data_type_idx == 0:
                ax.legend(loc='upper left', fontsize=6)


    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig_name = "%scompare_otu1_param.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




def plot_timescale_vs_phase():

    fig = plt.figure(figsize = (8.5, 8)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    max_coeff = 0.95

    for env_variable_idx, env_variable in enumerate(['doc', 'ph']):

        for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

            freq_all = []
            phase_all = []
            coeff_all = []

            for otu_label_idx, otu_label in enumerate(param_dict['otu_labels']):

                if otu_label_idx == 0:
                    continue

                param_otu = gam_coeff_dict[otu_label][data_type.lower()][env_variable]['coeff']
                freq_all.append(param_dict['freq_mle'][data_type][otu_label_idx])
                phase_all.append(param_dict['phase_mle'][data_type][otu_label_idx])
                coeff_all.append(param_otu)


            freq_all = numpy.asarray(freq_all)
            phase_all = numpy.asarray(phase_all)
            coeff_all = numpy.asarray(coeff_all)

            timescale_all = 2*numpy.pi/freq_all

            focal_otu_timescale =  2*numpy.pi/param_dict['freq_mle'][data_type][0]
            focal_otu_phase =  2*numpy.pi/param_dict['phase_mle'][data_type][0]
            focal_otu_coeff = gam_coeff_dict[param_dict['otu_labels'][0]][data_type.lower()][env_variable]['coeff']

            ax = fig.add_subplot(gs[env_variable_idx, data_type_idx])
            ax.scatter(focal_otu_timescale, focal_otu_phase, alpha=1, s=50, color='k', label='OTU 1 (phototroph)', zorder=2)

            ax.plot([0,focal_otu_timescale], [focal_otu_phase,focal_otu_phase], lw=2, ls=':', c='k', zorder=1)
            ax.plot([focal_otu_timescale,focal_otu_timescale], [0,focal_otu_phase], lw=2, ls=':', c='k', zorder=1)

            #edgecolor = utils.dna_rna_color_dict[data_type]
            #cmap='RdBu', norm=colors.Normalize(vmin=-1*lim_, vmax=lim_)       
            ax.scatter(timescale_all, phase_all, alpha=1, s=40, edgecolors='k', c=coeff_all, cmap='RdBu', norm=colors.Normalize(vmin=-1*max_coeff, vmax=max_coeff), label='Heterotrophic OTUs', zorder=2)

            ax.set_xlim([0, max(timescale_all)])

            phase_ticks = [0, 0.5*numpy.pi, numpy.pi, 1.5*numpy.pi, 2*numpy.pi]
            ax.set_ylim([min(phase_ticks), max(phase_ticks)])
            phase_tick_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
            ax.set_yticks(phase_ticks)
            ax.set_yticklabels(phase_tick_labels)
            ax.yaxis.set_tick_params(labelsize=7)

            ax.set_xlabel("Oscillation timescale (days), " + r'$\tau_{i}^{\mathrm{env}}$', fontsize=12)
            ax.set_ylabel("Phase, " +r'$\psi_{i}$', fontsize=12)


            if env_variable_idx == 0:
                ax.set_title(data_type, fontsize=14)

            if env_variable_idx + data_type_idx == 0:
                ax.legend(loc='upper left', fontsize=6)

            if data_type_idx == 0:
                ax.text(-0.22, 0.5, utils.env_variable_label_dict[env_variable], fontsize=14, ha='center', va='center', rotation=90, transform=ax.transAxes)


    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig_name = "%stimescale_vs_phase_for_coeff.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()





def plot_diff_timescale_vs_phase():

    fig = plt.figure(figsize = (8.5, 8)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=2, ncols=2)

    max_coeff = 0.95

    for env_variable_idx, env_variable in enumerate(['doc', 'ph']):

        for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

            freq_all = []
            phase_all = []
            coeff_all = []

            for otu_label_idx, otu_label in enumerate(param_dict['otu_labels']):

                if otu_label_idx == 0:
                    continue

                param_otu = gam_coeff_dict[otu_label][data_type.lower()][env_variable]['coeff']
                freq_all.append(param_dict['freq_mle'][data_type][otu_label_idx])
                phase_all.append(param_dict['phase_mle'][data_type][otu_label_idx])
                coeff_all.append(param_otu)

            freq_all = numpy.asarray(freq_all)
            phase_all = numpy.asarray(phase_all)
            coeff_all = numpy.asarray(coeff_all)

            timescale_all = 2*numpy.pi/freq_all

            focal_otu_timescale =  2*numpy.pi/param_dict['freq_mle'][data_type][0]
            focal_otu_phase =  2*numpy.pi/param_dict['phase_mle'][data_type][0]
            focal_otu_coeff = gam_coeff_dict[param_dict['otu_labels'][0]][data_type.lower()][env_variable]['coeff']

            diff_timescale_all = numpy.absolute(timescale_all - focal_otu_timescale)
            delta_phase = phase_all - focal_otu_phase
            # max delta can be +/- pi
            diff_phase_new = []
            for d in delta_phase:

                if d > numpy.pi:
                    diff_phase_new.append(d - 2*numpy.pi)
                elif d < -numpy.pi:
                    diff_phase_new.append(d + 2*numpy.pi)
                else:
                    diff_phase_new.append(d)


            diff_phase_new = numpy.asarray(diff_phase_new)
            diff_phase_new = numpy.absolute(diff_phase_new)

            ax = fig.add_subplot(gs[env_variable_idx, data_type_idx])

            #edgecolor = utils.dna_rna_color_dict[data_type]
            #cmap='RdBu', norm=colors.Normalize(vmin=-1*lim_, vmax=lim_)       
            ax.scatter(diff_timescale_all, diff_phase_new, alpha=1, s=40, edgecolors='k', c=coeff_all, cmap='RdBu', norm=colors.Normalize(vmin=-1*max_coeff, vmax=max_coeff), label='Heterotrophic OTUs', zorder=2)

            ax.set_xlim([0, max(timescale_all)])

            phase_ticks = [0, 0.25*numpy.pi, 0.5*numpy.pi, 0.75*numpy.pi, numpy.pi]
            ax.set_ylim([min(phase_ticks), max(phase_ticks)])
            phase_tick_labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
            ax.set_yticks(phase_ticks)
            ax.set_yticklabels(phase_tick_labels)
            ax.yaxis.set_tick_params(labelsize=7)

            ax.set_xlabel("Oscillation timescale diff. (days), " + r'$|\tau_{i}^{\mathrm{env}} - \tau_{1}^{\mathrm{env}}|$', fontsize=11)
            ax.set_ylabel("Phase difference, " +r'$|\psi_{i} - \psi_{1}|$', fontsize=12)


            if env_variable_idx == 0:
                ax.set_title(data_type, fontsize=14)

            if env_variable_idx + data_type_idx == 0:
                ax.legend(loc='upper left', fontsize=6)

            if data_type_idx == 0:
                ax.text(-0.22, 0.5, utils.env_variable_label_dict[env_variable], fontsize=14, ha='center', va='center', rotation=90, transform=ax.transAxes)


    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig_name = "%sdiff_timescale_vs_phase_for_coeff.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()



def plot_diff_timescale_vs_amp():

    fig = plt.figure(figsize = (8.5, 4)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=1, ncols=2)

    legend_elements = [Line2D([0], [0], marker='o',color='w', markeredgecolor='k', label='Heterotroph', markerfacecolor='w', markeredgewidth=1.4, markersize=8)]

    max_coeff = 0.95

    for env_variable_idx, env_variable in enumerate(['doc']):

        subplot_label_all = ['d', 'e']

        for data_type_idx, data_type in enumerate(['DNA', 'RNA']):

            freq_all = []
            phase_all = []
            coeff_all = []

            otu_label_het_all = []
            for otu_label_idx, otu_label in enumerate(param_dict['otu_labels']):

                if otu_label_idx == 0:
                    continue

                param_otu = gam_coeff_dict[otu_label][data_type.lower()][env_variable]['coeff']
                freq_all.append(param_dict['freq_mle'][data_type][otu_label_idx])
                phase_all.append(param_dict['amp_mle'][data_type][otu_label_idx])
                coeff_all.append(param_otu)

                otu_label_het_all.append(otu_label)


            freq_all = numpy.asarray(freq_all)
            phase_all = numpy.asarray(phase_all)
            coeff_all = numpy.asarray(coeff_all)

            timescale_all = 2*numpy.pi/freq_all

            focal_otu_timescale =  2*numpy.pi/param_dict['freq_mle'][data_type][0]
            #focal_otu_phase =  2*numpy.pi/param_dict['amp_mle'][data_type][0]
            #focal_otu_coeff = gam_coeff_dict[param_dict['otu_labels'][0]][data_type.lower()][env_variable]['coeff']

            diff_timescale_all = numpy.absolute(timescale_all - focal_otu_timescale)

            to_keep_idx = (diff_timescale_all/focal_otu_timescale) <= 0.05
            otu_label_het_all = numpy.asarray(otu_label_het_all)
            print(data_type, otu_label_het_all[to_keep_idx])

            #delta_phase = phase_all - focal_otu_phase
            # max delta can be +/- pi
            #diff_phase_new = []
            #for d in delta_phase:

            #    if d > numpy.pi:
            #        diff_phase_new.append(d - 2*numpy.pi)
            #    elif d < -numpy.pi:
            #        diff_phase_new.append(d + 2*numpy.pi)
            #    else:
            #        diff_phase_new.append(d)


            #diff_phase_new = numpy.asarray(diff_phase_new)
            #diff_phase_new = numpy.absolute(phase_all - focal_otu_phase)
            #diff_phase_new = focal_otu_phase - phase_all

            slope, intercept, r_value, p_value, std_err = stats.linregress(diff_timescale_all, phase_all)
            #print(slope, p_value)

            rho_2, p_value = utils.corr_permute_test(diff_timescale_all, phase_all)



            ax = fig.add_subplot(gs[env_variable_idx, data_type_idx])

            #edgecolor = utils.dna_rna_color_dict[data_type]
            #cmap='RdBu', norm=colors.Normalize(vmin=-1*lim_, vmax=lim_)       
            #ax.scatter(diff_timescale_all, phase_all, alpha=1, s=40, edgecolors='k', c=coeff_all, cmap='RdBu', norm=colors.Normalize(vmin=-1*max_coeff, vmax=max_coeff), zorder=2)
            ax.scatter(diff_timescale_all, phase_all, alpha=1, s=40, c=utils.dna_rna_color_dict[data_type], zorder=2)
            ax.set_xlim([0, max(timescale_all)])

            x_range = numpy.linspace(0, 400, num=1000)
            y_pred = intercept + (slope*x_range)
            ax.plot(x_range, y_pred, c='k', ls='--', lw=3, zorder=3)

            #print(min(coeff_all), max(coeff_all))

            #phase_ticks = [0, 0.25*numpy.pi, 0.5*numpy.pi, 0.75*numpy.pi, numpy.pi]
            #ax.set_ylim([min(phase_ticks), max(phase_ticks)])
            #phase_tick_labels = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
            #ax.set_yticks(phase_ticks)
            ##ax.set_yticklabels(phase_tick_labels)
            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)

            ax.set_xlabel("Timescale difference (days), " + r'$|\tau_{\mathrm{photo}}^{\mathrm{env}} - \tau_{i}^{\mathrm{env}}|$', fontsize=11)
            ax.set_ylabel("Amplitude, " +r'$A_{i}$', fontsize=12)

            ax.text(0.74, 0.84, r'$\rho^{2} = $' + str(round(rho_2, 4)), fontsize=10, ha='center', va='center', transform=ax.transAxes)
            ax.text(0.74, 0.76, r'$P = $' + str(round(p_value, 4)), fontsize=10, ha='center', va='center', transform=ax.transAxes)


            if data_type_idx == 0:
                ax.legend(handles=legend_elements, loc='upper right')


            if env_variable_idx == 0:
                ax.set_title(data_type, fontsize=14, color=utils.dna_rna_color_dict[data_type], fontweight='bold')

            #if env_variable_idx + data_type_idx == 0:
            #    ax.legend(loc='upper left', fontsize=6)

            #if data_type_idx == 0:
            #    ax.text(-0.22, 0.5, utils.env_variable_label_dict[env_variable], fontsize=14, ha='center', va='center', rotation=90, transform=ax.transAxes)

            ax.text(-0.095, 1.06, subplot_label_all[data_type_idx], fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)


    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig_name = "%sdiff_timescale_vs_amp_for_coeff.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")


    param_dict = pickle.load(open(sine_parameter_utils.param_otu_mle_dict_path, "rb"))
    gam_coeff_dict = utils.build_gam_coeff_dict()

    #print(param_dict['param_mean_mle']['DNA'])

    plot_diff_timescale_vs_amp()


    #plot_timescale_vs_phase()
    #plot_diff_timescale_vs_amp()

