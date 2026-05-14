import config
import numpy
import utils
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, colors

from scipy import stats

s_by_s, otu_labels, samples = utils.load_count_data()
metadata_dict = utils.build_metadata_dict()
sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
samples_dna = samples[(sample_type=='DNA')]
samples_rna = samples[(sample_type=='RNA')]

days_dna = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='DNA')]])
days_rna = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='RNA')]])

sort_idx_dna = numpy.argsort(days_dna)
sort_idx_rna = numpy.argsort(days_rna)

samples_dna_sort = samples_dna[sort_idx_dna]
samples_rna_sort = samples_rna[sort_idx_rna]

days_dna_sort = days_dna[sort_idx_dna]
days_rna_sort = days_rna[sort_idx_rna]

minor_days, major_days, major_labels = utils.get_seasonal_tick_labels()

sample_dna_idx = numpy.asarray([numpy.where(samples == s)[0] for s in samples_dna_sort])
sample_rna_idx = numpy.asarray([numpy.where(samples == s)[0] for s in samples_rna_sort])
##s_by_s_dna_all, s_by_s_rna_all, otu_labels_occupancy = utils.subset_s_by_s_occupancy(s_by_s, otu_labels, samples, min_occupancy=0)

s_by_s_dna = s_by_s[:,sample_dna_idx]
s_by_s_rna = s_by_s[:,sample_rna_idx]

n_reads_dna = numpy.sum(s_by_s_dna, axis=0)
n_reads_rna = numpy.sum(s_by_s_rna, axis=0)


#sample_type = numpy.asarray([metadata_dict[s]['sample_type'] for s in samples])
#days = numpy.asarray([metadata_dict[s]['day'] for s in samples[(sample_type=='DNA')]])

mean_dna = numpy.mean(n_reads_dna)
mean_rna = numpy.mean(n_reads_rna)

sem_dna = numpy.std(n_reads_dna)/numpy.sqrt(len(n_reads_dna))
sem_rna = numpy.std(n_reads_rna)/numpy.sqrt(len(n_reads_rna)) 

print("DNA mean #reads = %.2f +/- %.2f" % (mean_dna, sem_dna) )
print("RNA mean #reads = %.2f +/- %.2f" % (mean_rna, sem_rna) )

label_dna = 'DNA, ' + r'$\bar{N}_{\mathrm{DNA}} = $' + "{:,}".format(int(mean_dna)) + r'$\pm$' + str(int(sem_dna)) + ' (Mean ' + r'$\pm$' + ' SEM)'
label_rna = 'RNA, ' + r'$\bar{N}_{\mathrm{RNA}} = $' +"{:,}".format(int(mean_rna)) + r'$\pm$' + str(int(sem_rna)) + ' (Mean ' + r'$\pm$' + ' SEM)'




class OOMFormatter(matplotlib.ticker.ScalarFormatter):
   
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format



fig = plt.figure(figsize = (8, 4))
fig.subplots_adjust(bottom= 0.15)

ax = plt.subplot2grid((1, 1), (0, 0))

ax.set_xlim([0, max(days_dna)])
ax.set_xticks(minor_days, minor=True)
ax.set_xticks(major_days, minor=False)
ax.set_xticklabels(major_labels, minor=False, fontsize=7)
ax.yaxis.set_tick_params(labelsize=7)



        
ax.scatter(days_dna_sort, n_reads_dna, s=10, alpha=1, c=utils.dna_rna_color_dict['DNA'], label=label_dna)
ax.plot(days_dna_sort, n_reads_dna, ls='-', lw=1, c=utils.dna_rna_color_dict['DNA'])

ax.scatter(days_rna_sort, n_reads_rna, s=10, alpha=1, c=utils.dna_rna_color_dict['RNA'], label=label_rna)
ax.plot(days_rna_sort, n_reads_rna, ls='-', lw=1, c=utils.dna_rna_color_dict['RNA'])

ax.set_xlabel("Time (days)", fontsize=10)
ax.set_ylabel("Number of reads, " + r'$N(t)$', fontsize=10)

ax.legend(loc='upper left', fontsize=10)


yticks =  ax.get_yticks()
ytick_labels = ["{:,}".format(int(y)) for y in yticks]

print(yticks, ytick_labels)

ax.set_yticks(yticks[1:], minor=False)
ax.set_yticklabels(ytick_labels[1:], minor=False, fontsize=7)

#ax.yaxis.set_major_formatter(OOMFormatter(5, "%1.1f"))
#axe.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))



fig.subplots_adjust(hspace=0.35,wspace=0.3)
fig_name = "%sn_reads.png" % config.analysis_directory
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.4, dpi = 600)
plt.close()