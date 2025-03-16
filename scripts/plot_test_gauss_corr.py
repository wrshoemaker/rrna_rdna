
import numpy
import config
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from scipy import stats, signal
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


numpy.random.seed(123456789)
num_samples = 124
rng = numpy.random.default_rng()


fig = plt.figure(figsize = (4.5, 4)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=1, ncols=1)

ax = fig.add_subplot(gs[0, 0])

# The desired mean values of the sample.
mu = numpy.array([0, 0])
rho_all = [0, 0.1, 0.2, 0.3,  0.4, 0.5,  0.6, 0.7, 0.8, 0.9]
for rho in rho_all:

    # The desired covariance matrix.
    r = numpy.array([
            [1, rho],
            [rho, 1]])


    n_iter = 1000
    corr_all = []
    for i in range(n_iter):

        y = rng.multivariate_normal(mu, r, size=num_samples)

        diff = (y[:,0] - y[:,1])[:-1]
        delta_y1 = y[1:,0] - y[:-1,0]
        corr_all.append(numpy.corrcoef(diff, delta_y1)[0,1])


    ax.scatter(rho, numpy.mean(corr_all), c='k')

ax.set_xlabel('True correlation b/w Gaussian RVs X1 and X2')
ax.set_ylabel('Mean corr b/w X1 - X2 vs. change in X1')


ax.axhline(y=0, lw=2, ls=':', c='k')

    

fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig_name = "%stest_bivar.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()
