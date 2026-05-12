import config
import sys
import random
import argparse
import copy
import numpy
import utils
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy import stats, signal
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D



m_dict = utils.build_metadata_dict()
samples = list(m_dict.keys())

time = numpy.array([m_dict[s]['hours_since_midnight'] for s in samples], dtype=float) 
day_of_year = numpy.array([m_dict[s]['day_of_year'] for s in samples], dtype=float) 
to_keep_idx = ~numpy.isnan(time)
time_no_nan = time[to_keep_idx]
day_of_year_no_nan = day_of_year[to_keep_idx]



fig, ax = plt.subplots(figsize=(4.5,4))

ax.scatter(time_no_nan, day_of_year_no_nan, c='k')

slope, intercept, r_value, p_value, std_err = stats.linregress(time_no_nan, day_of_year_no_nan)


ax.text(0.26, 0.87, r'$\rho^{2} = $' + str(round(r_value**2, 3)), fontsize=12, ha='center', va='center', transform=ax.transAxes)
ax.text(0.26, 0.78, r'$P = $' + str(round(p_value, 4)), fontsize=12, ha='center', va='center', transform=ax.transAxes)

ax.set_xlabel("Time of sampling in a day (h)", fontsize=12)
ax.set_ylabel("Day of calendar year", fontsize=12)



fig.subplots_adjust(hspace=0.35, wspace=0.45)
fig_name = "%stime_vs_day.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()