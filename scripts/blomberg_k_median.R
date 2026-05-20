
rm(list = ls())
getwd()
setwd("~/GitHub/rrna_rdna/")


library(picante)
library(ape)
library(nlme)



n_perm <- 9999
set.seed(123456789)

lookup <- read.csv("data/gam_results/asv_id_lookup.csv")
asv_map <- unique(lookup[, c("asv_id", "base_name")])
asv_map$sequence <- sub("^ASV_", "", asv_map$base_name)
id_to_seq <- setNames(asv_map$sequence, asv_map$asv_id)

sens <- read.csv("data/gam_results/02_sensitivity_per_triplet.csv")
sig_preds <- read.csv("data/gam_results/sig_preds_rna_dna.csv")$predictor
tree <- read.tree("data/asv_w_outgroup_aligned_clean.fna.raxml.bestTree")

stopifnot(all(sens$asv_id %in% names(id_to_seq)))
stopifnot(all(id_to_seq %in% tree$tip.label))


dna <- sens[sens$dtype == "dna", c("asv_id", "predictor", "mean_abs_deriv")]
rna_dna <- sens[sens$dtype == "rna_dna", c("asv_id", "predictor", "mean_abs_deriv")]
merged <- merge(dna, rna_dna, by = c("asv_id", "predictor"), suffixes = c("_dna", "_rnadna"))
merged$diff <- merged$mean_abs_deriv_rnadna - merged$mean_abs_deriv_dna


k_results <- data.frame()
for (pred in sig_preds) {
  tryCatch({
    sub <- merged[merged$predictor == pred, ]
    d <- setNames(sub$diff, id_to_seq[sub$asv_id])
    pruned <- drop.tip(tree, setdiff(tree$tip.label, names(d)))
    d_ordered <- d[pruned$tip.label]
    k_result <- phylosignal(d_ordered, pruned)
    k_results <- rbind(k_results, data.frame(
      predictor = pred, K = round(k_result$K, 4), p_value = round(k_result$PIC.variance.P, 4), sig = k_result$PIC.variance.P < 0.05,
      interpretation = ifelse(
        k_result$PIC.variance.P >= 0.05,
        "no phylogenetic signal — Wilcoxon justified",
        "phylogenetic signal detected — run permutation test on median"
      )
    ))
  }, error = function(e) {
    cat("ERROR for", pred, ":", conditionMessage(e), "\n")
  })
}


print(k_results[, c("predictor","K","p_value","sig","interpretation")], row.names = FALSE)
