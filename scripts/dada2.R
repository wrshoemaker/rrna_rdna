#install.packages("devtools")
#library("devtools")
#devtools::install_github("benjjneb/dada2", ref="v1.26") # change the ref argument to get other versions

library(dada2); packageVersion("dada2")



path.fastq   <- "/Users/wrshoemaker/GitHub/rrna_rdna/data/fastq"
path.out     <- "/Users/wrshoemaker/GitHub/rrna_rdna/data/dada2"
path.meta    <- "/Users/wrshoemaker/GitHub/rrna_rdna/data/sra_metadata_annotated.csv"
silva.tax    <- "/Users/wrshoemaker/GitHub/aging_macroeco/data/silva_nr99_v138.2_toGenus_trainset.fa.gz"
silva.sp     <- "/Users/wrshoemaker/GitHub/aging_macroeco/data/silva_v138.2_assignSpecies.fa.gz"

dir.create(path.out, recursive = TRUE, showWarnings = FALSE)

meta_full <- read.csv(path.meta, header = TRUE)
# Keep first and last column, rename
meta <- meta_full[, c(1, ncol(meta_full))]
colnames(meta) <- c("sample", "type")


run_dada2 <- function(samples, lib.type) {
  
  cat(" Running DADA2 for:", lib.type, "\n")

  out.dir <- file.path(path.out, lib.type)
  filt.dir <- file.path(out.dir, "filtered")
  dir.create(filt.dir, recursive = TRUE, showWarnings = FALSE)
  
  fnFs <- sort(file.path(path.fastq, paste0(samples, "_1.fastq.gz")))
  fnRs <- sort(file.path(path.fastq, paste0(samples, "_2.fastq.gz")))
  
  # Keep only samples where both files exist
  exists <- file.exists(fnFs) & file.exists(fnRs)
  fnFs <- fnFs[exists]
  fnRs <- fnRs[exists]
  sample.names <- samples[exists]
  
  cat("Samples found:", length(sample.names), "/", length(samples), "\n")
  
  pdf(file.path(out.dir, "quality_profiles_raw.pdf"))
  print(plotQualityProfile(fnFs[1:min(4, length(fnFs))]))
  print(plotQualityProfile(fnRs[1:min(4, length(fnRs))]))
  dev.off()
  
  # cDNA may need slightly more lenient maxEE due to RT errors
  maxEE.val <- if (lib.type == "RNA") c(2, 5) else c(2, 5)
  
  filtFs <- file.path(filt.dir, paste0(sample.names, "_1_filt.fastq.gz"))
  filtRs <- file.path(filt.dir, paste0(sample.names, "_2_filt.fastq.gz"))
  names(filtFs) <- sample.names
  names(filtRs) <- sample.names
  
  # truncLen = c(230, 160) 
  # try
  out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs,
                       truncLen   = c(230, 200),
                       maxN       = 0,
                       maxEE      = c(2, 5),
                       truncQ     = 2,
                       rm.phix    = TRUE,
                       compress   = TRUE,
                       multithread = TRUE)
  
  write.table(out, file.path(out.dir, "filter_trim_summary.txt"), sep = "\t", quote = FALSE)
  cat("Filter and trim done\n")
  print(head(out))
  
  # Remove samples with no reads after filtering
  filtFs <- filtFs[file.exists(filtFs)]
  filtRs <- filtRs[file.exists(filtRs)]
  sample.names <- names(filtFs)
  
  errF <- learnErrors(filtFs, multithread = TRUE)
  errR <- learnErrors(filtRs, multithread = TRUE)
  
  pdf(file.path(out.dir, "error_profiles.pdf"))
  print(plotErrors(errF, nominalQ = TRUE))
  print(plotErrors(errR, nominalQ = TRUE))
  dev.off()
  
  dadaFs <- dada(filtFs, err = errF, multithread = TRUE, pool = TRUE)
  dadaRs <- dada(filtRs, err = errR, multithread = TRUE, pool = TRUE)
  
  mergers <- mergePairs(dadaFs, filtFs, dadaRs, filtRs, verbose = TRUE)
  
  seqtab <- makeSequenceTable(mergers)
  cat("Sequence length distribution:\n")
  print(table(nchar(getSequences(seqtab))))
  
  seqtab.nochim <- removeBimeraDenovo(seqtab, method = "consensus",
                                      multithread = TRUE, verbose = TRUE)
  cat("Chimera removal:", round(sum(seqtab.nochim) / sum(seqtab) * 100, 1), "% reads retained\n")
  
  # Track reads
  getN <- function(x) sum(getUniques(x))
  track <- cbind(out[rownames(out) %in% paste0(sample.names, "_1.fastq.gz"), ],
                 sapply(dadaFs, getN),
                 sapply(dadaRs, getN),
                 sapply(mergers, getN),
                 rowSums(seqtab.nochim))
  colnames(track) <- c("input", "filtered", "denoisedF", "denoisedR", "merged", "nonchim")
  rownames(track) <- sample.names
  write.table(track, file.path(out.dir, "read_tracking.txt"), sep = "\t", quote = FALSE)
  cat("Read tracking:\n")
  print(head(track))
  
  #Assign taxonomy
  taxa <- assignTaxonomy(seqtab.nochim, silva.tax, multithread = TRUE)
  taxa <- addSpecies(taxa, silva.sp)
  
  # Export
  write.table(t(seqtab.nochim),
              file.path(out.dir, paste0("seqtab_nochim_", lib.type, ".txt")),
              sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
  
  write.table(taxa,
              file.path(out.dir, paste0("taxa_", lib.type, ".txt")),
              sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
  
  saveRDS(seqtab.nochim, file.path(out.dir, paste0("seqtab_nochim_", lib.type, ".rds")))
  saveRDS(taxa,          file.path(out.dir, paste0("taxa_", lib.type, ".rds")))
  
  cat("Done:", lib.type, "— outputs saved to", out.dir, "\n")
  return(list(seqtab = seqtab.nochim, taxa = taxa, track = track))
}


dna.samples  <- meta$sample[meta$type == "DNA"]
rna.samples <- meta$sample[meta$type == "RNA"]

results.dna  <- run_dada2(dna.samples,  "DNA")
results.rna <- run_dada2(rna.samples, "RNA")
