library(dada2)
cat("DADA2 version:", as.character(packageVersion("dada2")), "\n")

# Paths
path.fastq <- "/leonardo/home/userexternal/wshoemak/fastq"
path.out   <- file.path(Sys.getenv("WORK"), "dada2", "RNA")
path.meta  <- "/leonardo/home/userexternal/wshoemak/sra_metadata_annotated.csv"
silva.tax  <- "/leonardo/home/userexternal/wshoemak/silva_nr99_v138.2_toGenus_trainset.fa.gz"
silva.sp   <- "/leonardo/home/userexternal/wshoemak/silva_v138.2_assignSpecies.fa.gz"

dir.create(path.out, recursive = TRUE, showWarnings = FALSE)

# Load metadata and get RNA samples
meta_full <- read.csv(path.meta, header = TRUE)
meta      <- meta_full[, c(1, ncol(meta_full))]
colnames(meta) <- c("sample", "type")

rna.samples <- meta$sample[meta$type == "RNA"]
cat("RNA samples found in metadata:", length(rna.samples), "\n")

# DADA2
filt.dir <- file.path(path.out, "filtered")
dir.create(filt.dir, recursive = TRUE, showWarnings = FALSE)

fnFs <- sort(file.path(path.fastq, paste0(rna.samples, "_1.fastq.gz")))
fnRs <- sort(file.path(path.fastq, paste0(rna.samples, "_2.fastq.gz")))

exists      <- file.exists(fnFs) & file.exists(fnRs)
fnFs        <- fnFs[exists]
fnRs        <- fnRs[exists]
sample.names <- rna.samples[exists]
cat("Samples with both F and R files:", length(sample.names), "/", length(rna.samples), "\n")

# Quality profiles
pdf(file.path(path.out, "quality_profiles_raw.pdf"))
print(plotQualityProfile(fnFs[1:min(4, length(fnFs))]))
print(plotQualityProfile(fnRs[1:min(4, length(fnRs))]))
dev.off()

# Filter and trim
filtFs <- file.path(filt.dir, paste0(sample.names, "_1_filt.fastq.gz"))
filtRs <- file.path(filt.dir, paste0(sample.names, "_2_filt.fastq.gz"))
names(filtFs) <- sample.names
names(filtRs) <- sample.names

out <- filterAndTrim(fnFs, filtFs, fnRs, filtRs,
                     truncLen    = c(230, 200),
                     maxN        = 0,
                     maxEE       = c(2, 5),
                     truncQ      = 2,
                     rm.phix     = TRUE,
                     compress    = TRUE,
                     multithread = TRUE)

write.table(out, file.path(path.out, "filter_trim_summary.txt"), sep = "\t", quote = FALSE)
cat("Filter and trim done\n")
print(head(out))

# Drop samples with no reads after filtering
filtFs      <- filtFs[file.exists(filtFs)]
filtRs      <- filtRs[file.exists(filtRs)]
sample.names <- names(filtFs)

# Learn errors
errF <- learnErrors(filtFs, multithread = TRUE)
errR <- learnErrors(filtRs, multithread = TRUE)

pdf(file.path(path.out, "error_profiles.pdf"))
print(plotErrors(errF, nominalQ = TRUE))
print(plotErrors(errR, nominalQ = TRUE))
dev.off()

# Denoise
dadaFs <- dada(filtFs, err = errF, multithread = TRUE, pool = TRUE)
dadaRs <- dada(filtRs, err = errR, multithread = TRUE, pool = TRUE)

# Merge
mergers <- mergePairs(dadaFs, filtFs, dadaRs, filtRs, verbose = TRUE)

# Sequence table
seqtab <- makeSequenceTable(mergers)
cat("Sequence length distribution:\n")
print(table(nchar(getSequences(seqtab))))

# Remove chimeras
seqtab.nochim <- removeBimeraDenovo(seqtab, method = "consensus",
                                    multithread = TRUE, verbose = TRUE)
cat("Chimera removal:", round(sum(seqtab.nochim) / sum(seqtab) * 100, 1), "% reads retained\n")

# Read tracking
getN  <- function(x) sum(getUniques(x))
track <- cbind(out[rownames(out) %in% paste0(sample.names, "_1.fastq.gz"), ],
               sapply(dadaFs,   getN),
               sapply(dadaRs,   getN),
               sapply(mergers,  getN),
               rowSums(seqtab.nochim))
colnames(track) <- c("input", "filtered", "denoisedF", "denoisedR", "merged", "nonchim")
rownames(track) <- sample.names
write.table(track, file.path(path.out, "read_tracking.txt"), sep = "\t", quote = FALSE)
cat("Read tracking:\n")
print(head(track))

# Taxonomy
taxa <- assignTaxonomy(seqtab.nochim, silva.tax, multithread = TRUE)
taxa <- addSpecies(taxa, silva.sp)

# Export
write.table(t(seqtab.nochim),
            file.path(path.out, "seqtab_nochim_RNA.txt"),
            sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
write.table(taxa,
            file.path(path.out, "taxa_RNA.txt"),
            sep = "\t", row.names = TRUE, col.names = TRUE, quote = FALSE)
saveRDS(seqtab.nochim, file.path(path.out, "seqtab_nochim_RNA.rds"))
saveRDS(taxa,          file.path(path.out, "taxa_RNA.rds"))

cat("Done — outputs saved to", path.out, "\n")
