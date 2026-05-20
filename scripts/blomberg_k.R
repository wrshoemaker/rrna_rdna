
rm(list = ls())
getwd()
setwd("~/GitHub/rrna_rdna/")


library(picante)
library(ape)
library(nlme)



# check whether phylogenetic corrections are needed 
tree <- read.tree("data/asv_w_outgroup_aligned_clean.fna.raxml.bestTree")
sens <- read.csv("data/gam_results/02_sensitivity_per_triplet.csv")
lookup <- read.csv("data/gam_results/asv_id_lookup.csv")
sig_preds <- read.csv("data/gam_results/sig_preds_rna_dna.csv")$predictor

asv_map <- unique(lookup[, c("asv_id", "base_name")])
asv_map$sequence <- sub("^ASV_", "", asv_map$base_name)
id_to_seq <- setNames(asv_map$sequence, asv_map$asv_id)

tree <- drop.tip(tree, "NC_005042_1_353331_354795_Prochlorococcus_marinus_subsp_marinus_str_CCMP1375_complete_genome")

stopifnot(all(sens$asv_id %in% names(id_to_seq)))
stopifnot(all(id_to_seq %in% tree$tip.label))


# sensitivity differences 
dna <- sens[sens$dtype == "dna", c("asv_id", "predictor", "mean_abs_deriv")]
rna_dna <- sens[sens$dtype == "rna_dna", c("asv_id", "predictor", "mean_abs_deriv")]
merged <- merge(dna, rna_dna, by = c("asv_id", "predictor"), suffixes = c("_dna", "_rnadna"))
merged$diff <- merged$mean_abs_deriv_rnadna - merged$mean_abs_deriv_dna




results <- data.frame()
for (pred in sig_preds) {
  tryCatch({
    sub <- merged[merged$predictor == pred, ]
    d   <- setNames(sub$diff, id_to_seq[sub$asv_id])
    
    pruned    <- drop.tip(tree, setdiff(tree$tip.label, names(d)))
    d_ordered <- d[pruned$tip.label]
    
    cat("\n--- Predictor:", pred, "---\n")
    cat("n ASVs:", length(d_ordered), "\n")
    cat("Any NA in d_ordered:", any(is.na(d_ordered)), "\n")
    cat("Any NA names:", any(is.na(names(d_ordered))), "\n")
    
    k_result <- phylosignal(d_ordered, pruned)
    
    results <- rbind(results, data.frame(
      predictor      = pred,
      K              = round(k_result$K, 4),
      p_value        = round(k_result$PIC.variance.P, 4),
      sig            = k_result$PIC.variance.P < 0.05,
      interpretation = ifelse(
        k_result$PIC.variance.P >= 0.05,
        "no phylogenetic signal — Wilcoxon justified",
        ifelse(k_result$K > 1,
               "K > 1: stronger signal than Brownian — consider PGLS",
               "K < 1: weaker signal than Brownian — consider PGLS")
      )
    ))
    cat("OK\n")
    
  }, error = function(e) {
    cat("ERROR for", pred, ":", conditionMessage(e), "\n")
  })
}



# phylo GLS
sub_wt <- merged[merged$predictor == "water_temp", ]
#d_wt <- setNames(sub_wt$diff, id_to_seq[sub_wt$asv_id])

df_wt <- data.frame(
  diff    = d_wt[pruned_wt$tip.label],
  species = pruned_wt$tip.label,         # explicit species column
  row.names = pruned_wt$tip.label
)

pruned_wt <- drop.tip(tree, setdiff(tree$tip.label, names(d_wt)))


fit_pgls <- gls(diff ~ 1,
                data        = df_wt,
                correlation = corBrownian(phy   = pruned_wt,
                                          form  = ~species),   # fix the warning
                method      = "ML")
summary(fit_pgls)

#df_wt <- data.frame(diff = d_wt[pruned_wt$tip.label], row.names = pruned_wt$tip.label)
#fit_pgls <- gls(diff ~ 1, data = df_wt, correlation = corBrownian(phy = pruned_wt), method = "ML")
#summary(fit_pgls)



  