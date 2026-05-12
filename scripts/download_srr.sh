


OUTDIR="/Users/wrshoemaker/GitHub/rrna_rdna/data/srr"
mkdir -p $OUTDIR

#while read SRR; do
#    prefetch $SRR && \
#    fasterq-dump $SRR \
#        --split-3 \
#        --threads 8 \
#        --outdir $OUTDIR && \
#    gzip $OUTDIR/${SRR}*.fastq && \
#    rm -rf $SRR   # remove .sra cache after conversion
#done < /Users/wrshoemaker/GitHub/rrna_rdna/data/SRR_Acc_List.txt


#cat  /Users/wrshoemaker/GitHub/rrna_rdna/data/SRR_Acc_List.txt | parallel -j 4 '
#    prefetch {} &&
#    fasterq-dump {} --split-3 --threads 4 --outdir '"$OUTDIR"' &&
#    gzip '"$OUTDIR"'/{}*.fastq &&
#    rm -rf {}'


ACC_LIST=/Users/wrshoemaker/GitHub/rrna_rdna/data/SRR_Acc_List.txt 

TOTAL=0
FOUND=0
MISSING=0
MISSING_LIST=()


while read -r SRR; do
    [[ -z "$SRR" ]] && continue  # skip empty lines
    ((TOTAL++))

    # Check for fastq.gz files matching the SRR accession
    FILES=$(find "$OUTDIR" -name "${SRR}*.fastq.gz" 2>/dev/null)

    if [[ -n "$FILES" ]]; then
        ((FOUND++))
    else
        ((MISSING++))
        MISSING_LIST+=("$SRR")
        echo "[MISSING] $SRR"
    fi

done < "$ACC_LIST"


rm -f SRR12672826/SRR12672826.sra.lock

prefetch SRR12672826 && \
fasterq-dump SRR12672826 --split-3 --threads 4 --outdir $OUTDIR && \
gzip $OUTDIR/SRR12672826*.fastq && \
rm -rf SRR12672826


# get metadata
# efetch -db sra -input SRR_Acc_List.txt -format runinfo > sra_metadata.csv


#awk -F',' 'BEGIN {OFS=","}
#    NR==1 {
#        print $0, "library_type"
#        for (i=1; i<=NF; i++) if ($i == "LibrarySource") col=i
#    }
#    NR>1 {
#        if ($col == "METATRANSCRIPTOMIC") print $0, "RNA"
#        else if ($col == "METAGENOMIC")   print $0, "DNA"
#        else                              print $0, "NA"
#    }
#' sra_metadata.csv > sra_metadata_annotated.csv