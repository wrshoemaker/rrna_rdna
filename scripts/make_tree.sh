#!/bin/bash

fasta_path="/Users/wrshoemaker/GitHub/rrna_rdna/data/asv_w_outgroup.fna"
alignment_path="/Users/wrshoemaker/GitHub/rrna_rdna/data/asv_w_outgroup_aligned.fna"



muscle -super5 ${fasta_path} -output ${alignment_path}

# clean alignment with python

#raxml-ng --all --msa ${asv_muscle} --msa-format FASTA --data-type DNA --seed 123456789 --model GTR+G --bs-trees autoMRE -outgroup NC_005042_1_353331_354795_Prochlorococcus_marinus_subsp_marinus_str_CCMP1375_complete_genome

#FastTree -nt -gtr -gamma ${asv_muscle} > ${tree}