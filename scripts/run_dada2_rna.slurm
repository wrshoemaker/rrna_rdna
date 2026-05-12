#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=20:00:00
#SBATCH --error=/leonardo/home/userexternal/wshoemak/dada2_rna.err
#SBATCH --output=/leonardo/home/userexternal/wshoemak/dada2_rna.out
#SBATCH --account=ICT24_CMSP_CPU
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=64G

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate dada2_env

# cp -r /Users/wrshoemaker/GitHub/rrna_rdna/data/fastq wshoemak@login.leonardo.cineca.it:/leonardo/home/userexternal/wshoemak

Rscript /leonardo/home/userexternal/wshoemak/dada2_rna.R
