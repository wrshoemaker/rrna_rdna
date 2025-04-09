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

# Fig. 4, S15
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

# Fig. S4, S6, S7 compare_clr_to_true_abundance_oscillating
python simulation_utils.py

# Fig. S5 clr_vs_rel_abund_comparison_formatted
python plot_clr_vs_rel_abund_comparison()

# Fig. S8
pyhton plot_n_reads.py

# Figs. S9, S10, S16; time_vs_abundance_clr_DNA, time_vs_abundance_clr_RNA, time_vs_clr_ratio
python sine_parameter_utils.py

# Figs. S11, S12; autocorrelation_otu_DNA, autocorrelation_otu_RNA
python plot_autocorrelation_otu.py

# Fig. S13; phylo_dist_vs_sine_params
python plot_phylo_dist_vs_sine_params.py

# Fig. S14; data_collapse_simulation
python plot_data_collapse_simulation.py

# Fig. S15; predict_change_dna.png
python plot_predict_change_dna.py

# 



#python python plot_clr_comparison.py
#python plot_clr_vs_rel_abund_comparison.py
#python plot_clr_vs_clr_pseudo_comparison.py







