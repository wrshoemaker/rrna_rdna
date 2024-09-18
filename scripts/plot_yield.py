
import numpy
import pandas

import config

from scipy import stats, signal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# https://bionumbers.hms.harvard.edu/files/Yield%20characteristics%20of%20E.%20coli%20B%20growing%20in%20glucose-limited%20aerobic%20continuous%20culture%20over%20the%20temperature%20range%2017.5%20to%2042.0%20C.pdf

df = pandas.read_csv('%syield_chemostat_bionumbers.csv' % config.data_directory, sep=',')

temp_all = list(set(df['Temperature (C)'].tolist()))
temp_all.sort()

rgb = cm.Reds(numpy.linspace(0,1,len(temp_all)+3))
rgb = mpl.colors.ListedColormap(rgb[int(0.2*len(temp_all)):,:-1])


def plot_yield():

    fig = plt.figure(figsize = (8.5, 8)) #
    fig.subplots_adjust(bottom= 0.1, wspace=0.1)

    ax_dryweight = plt.subplot2grid((2, 2), (0, 0))
    ax_output = plt.subplot2grid((2, 2), (0, 1))
    ax_respiration = plt.subplot2grid((2, 2), (1, 0))
    ax_yield = plt.subplot2grid((2, 2), (1, 1))

    
    for t_idx, t in enumerate(temp_all):

        df_t = df[df['Temperature (C)'] == t]

        dilution_t = df_t['Dilution (h^-1)'].values
        dryweight_t = df_t['Dry weight (ug/mL)'].values
        output_t = df_t['Output (mg dry weight per-hr)'].values
        respiration_t = df_t['Respiration (mmol O2 per-hour)'].values
        yield_t = df_t['Yield (g dry weight per g-atom O)'].values

        #ax_dryweight.scatter(dilution_t, dryweight_t)
        ax_dryweight.scatter(dilution_t, dryweight_t, s=30, color=rgb(t_idx), label='%.1f C' % t, zorder=2)
        ax_dryweight.plot(dilution_t, dryweight_t, ls='-', lw=2, color=rgb(t_idx), alpha=0.5, zorder=1)

        ax_output.scatter(dilution_t, output_t, s=30, color=rgb(t_idx), label='%.1f C' % t, zorder=2)
        ax_output.plot(dilution_t, output_t, ls='-', lw=2, color=rgb(t_idx), alpha=0.5, zorder=1)

        ax_respiration.scatter(dilution_t, respiration_t, s=30, color=rgb(t_idx), label='%.1f C' % t, zorder=2)
        ax_respiration.plot(dilution_t, respiration_t, ls='-', lw=2, color=rgb(t_idx), alpha=0.5, zorder=1)


        ax_yield.scatter(dilution_t, yield_t, s=30, color=rgb(t_idx), label='%f C' % t, zorder=2)
        ax_yield.plot(dilution_t, yield_t, ls='-', lw=2, color=rgb(t_idx), alpha=0.5, zorder=1)


    ax_dryweight.legend(loc='lower left', fontsize=8)

    ax_dryweight.set_xlabel('Dilution rate (h^-1)', fontsize=10)
    ax_dryweight.set_ylabel('Dry weight (ug/mL)', fontsize=10)

    ax_output.set_xlabel('Dilution rate (h^-1)', fontsize=10)
    ax_output.set_ylabel('Output (mg dry weight per-hr)', fontsize=10)

    ax_respiration.set_xlabel('Dilution rate (h^-1)', fontsize=10)
    ax_respiration.set_ylabel('Respiration (mmol O2 per-hour)', fontsize=10)

    ax_yield.set_xlabel('Dilution rate (h^-1)', fontsize=10)
    ax_yield.set_ylabel('Yield (g dry weight per g atom O)', fontsize=10)

    fig.subplots_adjust(hspace=0.25,wspace=0.25)
    fig_name = "%syield.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_yield_coefficients(max_dilution=0.35):

    fig = plt.figure(figsize = (8.5, 4)) #
    fig.subplots_adjust(bottom= 0.1, wspace=0.15)

    ax_slope = plt.subplot2grid((1, 2), (0, 0))
    ax_intercept = plt.subplot2grid((1, 2), (0, 1))

    slope_all = []
    intercept_all = []
    for t_idx, t in enumerate(temp_all):

        df_t = df[df['Temperature (C)'] == t]

        dilution_t = df_t['Dilution (h^-1)'].values
        yield_t = df_t['Yield (g dry weight per g-atom O)'].values

        to_keep_idx = dilution_t <= max_dilution

        dilution_t = dilution_t[to_keep_idx]
        yield_t = yield_t[to_keep_idx]

        slope, intercept, r_value, p_value, std_err = stats.linregress(dilution_t, yield_t)

        # https://github.com/scipy/scipy/blob/a3ffdface6d8779ffd91f605e4e102a9fda65a7f/scipy/stats/_stats_py.py
        # Line 10366
        ssxm, ssxym, _, ssym = numpy.cov(dilution_t, yield_t, bias=1).flat
        intercept_stderr = std_err * numpy.sqrt(ssxm + numpy.mean(dilution_t)**2)


        ax_slope.scatter(t, slope, s=30, color=rgb(t_idx), zorder=2)
        ax_intercept.scatter(t, intercept, s=30, color=rgb(t_idx), zorder=2)


        ax_slope.errorbar(t, slope, yerr=std_err, linestyle='-', marker='o', c='k', elinewidth=2, alpha=1, zorder=1)
        ax_intercept.errorbar(t, intercept, yerr=intercept_stderr, linestyle='-', marker='o', c='k', elinewidth=2, alpha=1, zorder=1)

        slope_all.append(slope)
        intercept_all.append(intercept)


    ax_slope.plot(temp_all, slope_all, c='k', lw=2, alpha=0.8)
    ax_intercept.plot(temp_all, intercept_all, c='k', lw=2, alpha=0.8)


    ax_slope.set_xlabel('Temperature (C)', fontsize=12)
    ax_slope.set_ylabel('Slope b/w dilution rate and yield', fontsize=12)

    ax_intercept.set_xlabel('Temperature (C)', fontsize=12)
    ax_intercept.set_ylabel('Intercept b/w dilution rate and yield', fontsize=12)


    fig.subplots_adjust(hspace=0.45,wspace=0.35)
    fig_name = "%syield_coefficients.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_rescaled_yield():

    fig = plt.figure(figsize = (4.5, 4)) #
    fig.subplots_adjust(bottom= 0.1, wspace=0.1)

    ax = plt.subplot2grid((1, 1), (0, 0))

    
    for t_idx, t in enumerate(temp_all):

        df_t = df[df['Temperature (C)'] == t]

        dilution_t = df_t['Dilution (h^-1)'].values
        dryweight_t = df_t['Dry weight (ug/mL)'].values
        output_t = df_t['Output (mg dry weight per-hr)'].values
        respiration_t = df_t['Respiration (mmol O2 per-hour)'].values
        yield_t = df_t['Yield (g dry weight per g-atom O)'].values

        rescaled_x = dryweight_t*dilution_t/respiration_t

        #ax.scatter(rescaled_x, yield_t, s=30, color=rgb(t_idx), label='%.1f C' % t, zorder=2)
        #ax.plot(rescaled_x, yield_t, ls='-', lw=2, color=rgb(t_idx), alpha=0.5, zorder=1)

        std_error = numpy.std(yield_t)/numpy.sqrt(len(yield_t))

        ax.errorbar(t, numpy.mean(yield_t), yerr=std_error, linestyle='-', marker='o', c='k', elinewidth=2, alpha=1, zorder=1)


        ax.scatter(t, numpy.mean(yield_t), s=30, color=rgb(t_idx), label='%.1f C' % t, zorder=2)


    #ax.set_xlabel('Dryweight * dilution / respiration, ug/mL*mmol', fontsize=12)
    #ax.set_ylabel('Yield (g dry weight per g-atom O)', fontsize=12)

    ax.set_xlabel('Temperature (C)', fontsize=12)
    ax.set_ylabel('Yield (g dry weight per g-atom O)', fontsize=12)

    fig.subplots_adjust(hspace=0.45,wspace=0.35)
    fig_name = "%srescaled_yield.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()
        



plot_rescaled_yield()

#plot_yield()

#plot_yield_coefficients()
