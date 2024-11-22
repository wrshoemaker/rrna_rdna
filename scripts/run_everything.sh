#!/bin/bash


# activate conda environment
source activate Py38


# run simulations



# Main figures

# Fig. 1
python plot_fig_1.py

# Fig. 2
python plot_fig_2.py

# Fig. 3
python plot_fig_3.py

# Fig. 4
python plot_fig_4.p

# Fig. 5
python plot_fig_5.py


# Supplemental figures

# Fig. S1
python plot_logfold_ratio.py

# Fig. S2
python plot_macroeco_summary.py

# Fig. S3 
python plot_compare_rna_dna_macroeco.py


# Fig. SX
python python plot_clr_comparison.py




pyhton plot_n_reads.py



# figs to cite
#time_vs_clr_ratio.png
# predict_change_dna.png
# fig6.png
# copy_number_vs_amp.png
# fig5_formatted.png
# time_vs_env.png

