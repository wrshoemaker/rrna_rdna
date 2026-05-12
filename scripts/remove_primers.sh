

# samples
# DRR422468
# DRR423490
# had a very small size for one of the files. Removed

#conda create -n cutadapt_env -c bioconda cutadapt
#conda activate cutadapt_env



INPUT="/Users/wrshoemaker/GitHub/rrna_rdna/data/fastq"
OUTPUT="/Users/wrshoemaker/GitHub/rrna_rdna/data/fastq_trimmed"



mkdir -p "$OUTPUT"

#for f1 in "$INPUT"/*_1.fastq.gz
#do
#    sample=$(basename "$f1" _1.fastq.gz)
#    f2="$INPUT/${sample}_2.fastq.gz"

#    if [[ -f "$f2" ]]; then
#        echo "Processing $sample"

#        cutadapt -j 8 \
#            -g GTGYCAGCMGCCGCGGTAA \
#            -G GGACTACNVGGGTWTCTAAT \
#            -a ATTAGAWACCCBNGTAGTCC \
#            -A TTACCGCGGCKGCTGRCAC \
#            --discard-untrimmed \
#            --pair-filter=any \
#            -o "$OUTPUT/${sample}_1.fastq.gz" \
#            -p "$OUTPUT/${sample}_2.fastq.gz" \
#            "$f1" "$f2"

#    else
#        echo "WARNING: missing reverse read for $sample"
#    fi
#done



f1=/Users/wrshoemaker/GitHub/rrna_rdna/data/fastq/SRR12672799_1.fastq.gz
f2=/Users/wrshoemaker/GitHub/rrna_rdna/data/fastq/SRR12672799_2.fastq.gz

cutadapt -j 8 \
            -g GTGYCAGCMGCCGCGGTAA \
            -G GGACTACNVGGGTWTCTAAT \
            -a ATTAGAWACCCBNGTAGTCC \
            -A TTACCGCGGCKGCTGRCAC \
            --discard-untrimmed \
            --pair-filter=any \
            -o "$OUTPUT/SRR12672799_1.fastq.gz" \
            -p "$OUTPUT/SRR12672799_2.fastq.gz" \
            "$f1" "$f2"



# check overrepresented sequences

gzcat ./SRR12673012_1.fastq.gz | awk 'NR%4==2' | head -1000 | cut -c1-30 > starts.txt
sort starts.txt | uniq -c | sort -rn | head -20
