#GAM Sensitivity Comparison — 21 ASV Triplets (_dna / _rna / _rna_dna)

# Pipeline
# 1. Read & transpose data -> samples x variables
# 2. Detect 21 triplets from _dna / _rna / _rna_dna suffixes
# 3. Fit joint GAM per triplet (response x smooth interaction)
# 4. Extract omnibus tests, sensitivity indices, difference smooth curves,
#    pointwise derivative curves (NEW for Option B gradient shape)
# 5. Apply BH-FDR correction jointly across all 693 tests
# 6. Save ALL figure-ready flat files
# 7. Run diagnostics: k-check, concurvity, deviance explained, near-zero EDF

rm(list = ls())
getwd()
setwd("~/GitHub/rrna_rdna/")

library(mgcv)
library(gratia)
library(dplyr)
library(tidyr)
library(ggplot2)

if (!requireNamespace("gratia", quietly = TRUE)) {
  stop("gratia is not installed. Run: install.packages('gratia')")
}
cat("gratia version:", as.character(packageVersion("gratia")), "\n")

DATA_PATH  <- "data/data_for_gam.csv"
OUTPUT_DIR <- "data/gam_results"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

env_predictors  <- c(
  "water_temp", "specific_conductivity", "salinity",
  "total_nitrogen", "total_phosphorus", "doc",
  "secchi_depth", "ph", "dissolved_oxygen"
)
temp_predictors <- c("day_of_year", "days")
all_predictors  <- c(env_predictors, temp_predictors)

dtype_labels <- c(dna = "DNA", rna = "RNA", rna_dna = "RNA:DNA")

raw <- read.csv(DATA_PATH, check.names = FALSE)
row_names <- raw$sample_number
mat <- as.matrix(raw[, -1])
df  <- as.data.frame(t(mat))
colnames(df) <- row_names
df <- df %>% mutate(across(everything(), as.numeric))

all_cols <- colnames(df)
asv_cols <- all_cols[grepl("^ASV_", all_cols)]

get_base  <- function(x) {
  x <- sub("_rna_dna$", "", x)
  x <- sub("_rna$", "", x)
  x <- sub("_dna$", "", x)
  x
}
get_dtype <- function(x) {
  ifelse(grepl("_rna_dna$", x), "rna_dna",
         ifelse(grepl("_rna$", x), "rna", "dna"))
}

asv_df <- data.frame(
  full_name = asv_cols,
  base_name = get_base(asv_cols),
  dtype     = get_dtype(asv_cols),
  stringsAsFactors = FALSE
)

base_names <- unique(asv_df$base_name)
n_triplets <- length(base_names)
asv_ids <- sprintf("ASV_%03d", seq_len(n_triplets))
id_map <- setNames(asv_ids, base_names)
asv_df$asv_id <- id_map[asv_df$base_name]
asv_df$short_col <- paste(asv_df$asv_id, asv_df$dtype, sep = "_")

write.csv(asv_df, file.path(OUTPUT_DIR, "asv_id_lookup.csv"), row.names = FALSE)

triplets <- lapply(base_names, function(base) {
  sub <- asv_df[asv_df$base_name == base, ]
  setNames(sub$full_name, sub$dtype)
})
names(triplets) <- asv_ids

incomplete <- sapply(triplets, length) != 3
if (any(incomplete)) {
  warning("Incomplete triplets: ", paste(names(which(incomplete)), collapse = ", "))
  triplets <- triplets[!incomplete]
}



# Model fitting
make_long <- function(df, triplet) {
  bind_rows(
    df %>% mutate(y = .data[[triplet[["dna"]]]], dtype = "dna"),
    df %>% mutate(y = .data[[triplet[["rna"]]]], dtype = "rna"),
    df %>% mutate(y = .data[[triplet[["rna_dna"]]]], dtype = "rna_dna")
  ) %>%
    mutate(dtype = factor(dtype, levels = c("dna", "rna", "rna_dna")))
}

fit_joint_gam <- function(df_long) {
  gam(
    y ~ dtype
    + s(water_temp, k = 5,  by = dtype)
    + s(specific_conductivity, k = 5,  by = dtype)
    + s(salinity, k = 5,  by = dtype)
    + s(total_nitrogen, k = 5,  by = dtype)
    + s(total_phosphorus, k = 5,  by = dtype)
    + s(doc, k = 5,  by = dtype)
    + s(secchi_depth, k = 5,  by = dtype)
    + s(ph, k = 5,  by = dtype)
    + s(dissolved_oxygen, k = 5,  by = dtype)
    + s(day_of_year, k = 12, bs = "cc", by = dtype)
    + s(days, k = 10, bs = "tp", by = dtype),
    data = df_long, method = "REML"
  )
}

fit_constrained_gam <- function(df_long) {
  gam(
    y ~ dtype
    + s(water_temp, k = 5)
    + s(specific_conductivity, k = 5)
    + s(salinity, k = 5)
    + s(total_nitrogen, k = 5)
    + s(total_phosphorus, k = 5)
    + s(doc, k = 5)
    + s(secchi_depth, k = 5)
    + s(ph, k = 5)
    + s(dissolved_oxygen, k = 5)
    + s(day_of_year, k = 12, bs = "cc")
    + s(days, k = 10, bs = "tp"),
    data = df_long, method = "REML"
  )
}



# Extraction functions
extract_omnibus <- function(m_joint, m_constrained, asv_id) {
  lrt <- anova(m_constrained, m_joint, test = "Chisq")
  data.frame(
    asv_id = asv_id,
    full_name = base_names[asv_ids == asv_id],
    delta_dev = lrt[2, "Deviance"],
    df = lrt[2, "Df"],
    p_raw = lrt[2, "Pr(>Chi)"],
    delta_AIC = AIC(m_constrained) - AIC(m_joint),
    dev_expl = summary(m_joint)$dev.expl
  )
}

extract_sensitivity <- function(m_joint, asv_id) {
  results <- list()
  smooth_names <- gratia::smooths(m_joint)

  for (pred in all_predictors) {
    for (dt in c("dna", "rna", "rna_dna")) {
      candidate <- paste0("s(", pred, "):dtype", dt)
      if (!any(grepl(candidate, smooth_names, fixed = TRUE))) {
        message(sprintf("  Term not found: %s [skipping]", candidate))
        next
      }

      d <- tryCatch(
        gratia::derivatives(m_joint, select = candidate, type = "central",
                            interval = "simultaneous", unconditional = TRUE),
        error   = function(e) {
          message(sprintf("  derivatives() error %s|%s|%s: %s",
                          asv_id, pred, dt, conditionMessage(e))); NULL },
        warning = function(w) {
          message(sprintf("  derivatives() warning %s|%s|%s: %s",
                          asv_id, pred, dt, conditionMessage(w))); NULL }
      )
      if (is.null(d) || nrow(d) == 0) next

      results[[length(results) + 1]] <- data.frame(
        asv_id = asv_id,
        dtype = dt,
        predictor = pred,
        mean_abs_deriv = mean(abs(d$.derivative)),
        sd_abs_deriv = sd(abs(d$.derivative)),
        n_points = nrow(d),
        prop_sig_region = mean((d$.lower_ci > 0) | (d$.upper_ci < 0))
      )
    }
  }
  bind_rows(results)
}



# Saves the full pointwise SIGNED derivative per ASV x dtype x predictor.
# Required for Option B gradient shape (derivative differences):
# delta_{j,i}(x) = d f_alt/dx(x) - d f_rDNA/dx(x)

# Columns: asv_id, dtype, predictor, x_value, derivative, lower_ci, upper_ci
# Note: derivative is the SIGNED first derivative, NOT abs value.
# 21 ASVs x 3 dtypes x 11 predictors x 100 pts ~ 69,300 rows.
extract_derivative_curves <- function(m_joint, asv_id) {
  results <- list()
  smooth_names <- gratia::smooths(m_joint)

  for (pred in all_predictors) {
    for (dt in c("dna", "rna", "rna_dna")) {
      candidate <- paste0("s(", pred, "):dtype", dt)
      if (!any(grepl(candidate, smooth_names, fixed = TRUE))) next

      d <- tryCatch(
        gratia::derivatives(m_joint, select = candidate, type = "central",
                            interval = "simultaneous", unconditional = TRUE),
        error   = function(e) {
          message(sprintf("  deriv_curves error %s|%s|%s: %s",
                          asv_id, pred, dt, conditionMessage(e))); NULL },
        warning = function(w) NULL
      )
      if (is.null(d) || nrow(d) == 0) next

      # Identify x-column — predictor name appears as column in gratia output
      x_col <- intersect(all_predictors, colnames(d))
      if (length(x_col) == 0) next

      results[[length(results) + 1]] <- data.frame(
        asv_id = asv_id,
        dtype = dt,
        predictor = pred,
        x_value = d[[x_col[1]]],
        derivative = d$.derivative,   # signed — NOT abs value
        lower_ci = d$.lower_ci,
        upper_ci = d$.upper_ci
      )
    }
  }
  bind_rows(results)
}


extract_diff_smooth_curves <- function(m_joint, asv_id) {
  pairs <- list(c("dna","rna"), c("dna","rna_dna"), c("rna","rna_dna"))
  results <- list()

  find_col <- function(d, candidates) {
    hit <- candidates[candidates %in% colnames(d)]
    if (length(hit) == 0) return(NA_character_)
    hit[1]
  }

  for (pred in all_predictors) {
    smooth_term <- paste0("s(", pred, ")")

    for (pair in pairs) {
      d <- tryCatch(
        gratia::difference_smooths(m_joint, smooth = smooth_term, levels = pair,
                                   ci_level = 0.95, unconditional = TRUE),
        error = function(e) {
          message(sprintf("  diff_smooth error %s|%s|%s vs %s: %s",
                          asv_id, pred, pair[1], pair[2], conditionMessage(e)))
          NULL }
      )
      if (is.null(d) || nrow(d) == 0) next

      d <- d[d$.level_1 == pair[1] & d$.level_2 == pair[2], ]
      if (nrow(d) == 0) next

      col_diff <- find_col(d, c(".diff", "difference", ".difference"))
      col_lower <- find_col(d, c(".lower_ci", "lower"))
      col_upper <- find_col(d, c(".upper_ci", "upper"))

      if (any(is.na(c(col_diff, col_lower, col_upper)))) {
        message(sprintf("  Cannot map diff/lower/upper for %s|%s. Columns: %s",
                        asv_id, pred, paste(colnames(d), collapse = ", ")))
        next
      }

      known_non_x <- c(col_diff, col_lower, col_upper,
                       ".smooth", ".by", ".level_1", ".level_2", ".se")
      x_col <- intersect(all_predictors, colnames(d))
      if (length(x_col) == 0) {
        leftover <- setdiff(colnames(d), known_non_x)
        x_col <- leftover[sapply(d[leftover], is.numeric)][1]
      }
      if (length(x_col) == 0 || is.na(x_col)) {
        message(sprintf("  Cannot identify x column for %s|%s", asv_id, pred))
        next
      }

      out <- d %>%
        rename(x_value = all_of(x_col[[1]]),
               difference = all_of(col_diff),
               lower = all_of(col_lower),
               upper = all_of(col_upper)) %>%
        mutate(asv_id = asv_id,
               predictor = pred,
               dtype_a = pair[1],
               dtype_b = pair[2],
               pair_label = paste(dtype_labels[pair[1]], "vs",
                                  dtype_labels[pair[2]])) %>%
        select(asv_id, predictor, dtype_a, dtype_b, pair_label,
               x_value, difference, lower, upper)

      results[[length(results) + 1]] <- out
    }
  }
  bind_rows(results)
}

summarise_diff_smooth <- function(curve_data) {
  curve_data %>%
    group_by(asv_id, predictor, dtype_a, dtype_b, pair_label) %>%
    summarise(
      prop_sig = mean((lower > 0) | (upper < 0)),
      net_direction = mean(difference),
      max_abs_diff = max(abs(difference)),
      .groups = "drop"
    )
}


# =============================================================================
# Diagnostics
# =============================================================================

extract_diagnostics <- function(m_joint, asv_id) {
  tryCatch({

    kchk <- tryCatch(gratia::k_check(m_joint), error = function(e) NULL)
    n_k_fail <- if (!is.null(kchk))
      sum(kchk$p.value < 0.05 & kchk[["k-index"]] < 1, na.rm = TRUE) else NA
    k_fail_terms <- if (!is.na(n_k_fail) && n_k_fail > 0)
      paste(rownames(kchk)[kchk$p.value < 0.05 & kchk[["k-index"]] < 1],
            collapse = "; ") else ""

    conc <- tryCatch(mgcv::concurvity(m_joint, full = TRUE),
                     error = function(e) NULL)
    max_concurvity<- if (!is.null(conc))
      round(max(conc["worst", ], na.rm = TRUE), 3) else NA
    conc_high_terms <- if (!is.null(conc)) {
      worst <- conc["worst", ]
      paste(names(worst)[!is.na(worst) & worst > 0.8], collapse = "; ")
    } else ""

    dev_expl <- round(summary(m_joint)$dev.expl * 100, 1)
    sm_tab <- summary(m_joint)$s.table
    near_zero_idx <- sm_tab[, "edf"] < 1.1
    n_near_zero <- sum(near_zero_idx, na.rm = TRUE)
    near_zero_terms <- if (n_near_zero > 0)
      paste(rownames(sm_tab)[near_zero_idx], collapse = "; ") else ""

    data.frame(
      asv_id = asv_id,
      dev_expl_pct = dev_expl,
      AIC = round(AIC(m_joint), 1),
      n_k_fail = n_k_fail,
      k_fail_terms = k_fail_terms,
      max_concurvity = max_concurvity,
      concurvity_flag = !is.na(max_concurvity) && max_concurvity > 0.8,
      conc_high_terms = conc_high_terms,
      n_near_zero_edf = n_near_zero,
      near_zero_terms = near_zero_terms,
      any_flag = ((!is.na(n_k_fail) && n_k_fail > 0) ||
                         (!is.na(max_concurvity) && max_concurvity > 0.8) ||
                         dev_expl < 20),
      stringsAsFactors = FALSE
    )

  }, error = function(e) {
    message(sprintf("  DIAG ERROR [%s]: %s", asv_id, conditionMessage(e)))
    data.frame(asv_id = asv_id, dev_expl_pct = NA, AIC = NA,
               n_k_fail = NA, k_fail_terms = "",
               max_concurvity = NA, concurvity_flag = NA, conc_high_terms = "",
               n_near_zero_edf = NA, near_zero_terms = "",
               any_flag = NA, stringsAsFactors = FALSE)
  })
}



# test one ASV
asv_check <- grep("^ASV_", colnames(df), value = TRUE)
cat("ASV columns found in df:", length(asv_check), "\n")
cat("First ASV column name:\n ", asv_check[1], "\n")

cat("\nTriplet indexing test:\n")
cat("Single bracket  class:", class(triplets[[1]]["dna"]),   "\n")
cat("Double bracket  class:", class(triplets[[1]][["dna"]]), "\n")

cat("\nTesting make_long on ASV_001...\n")
test_long <- tryCatch(make_long(df, triplets[[1]]),
                      error = function(e) { cat("make_long error:", conditionMessage(e), "\n"); NULL })
if (!is.null(test_long))
  cat("  OK — rows:", nrow(test_long), "| dtype levels:", levels(test_long$dtype), "\n")

cat("\nTesting fit_joint_gam on ASV_001...\n")
test_gam <- tryCatch(fit_joint_gam(test_long),
                     error   = function(e) { cat("GAM error:",   conditionMessage(e), "\n"); NULL },
                     warning = function(w) { cat("GAM warning:", conditionMessage(w), "\n"); NULL })
if (!is.null(test_gam))
  cat("  OK — deviance explained:", round(summary(test_gam)$dev.expl * 100, 1), "%\n")

test_diff <- gratia::difference_smooths(test_gam, smooth = "s(water_temp)",
                                        levels = c("dna","rna"), ci_level = 0.95)
cat("Columns returned by difference_smooths():\n"); print(colnames(test_diff))
cat("\nHead:\n"); print(head(test_diff, 3))

if (!is.null(test_gam)) {
  cat("\nTesting derivatives() on ASV_001 / water_temp / dna...\n")
  test_deriv <- tryCatch(
    gratia::derivatives(test_gam, select = "s(water_temp):dtypedna",
                        type = "central", interval = "simultaneous",
                        unconditional = TRUE),
    error   = function(e) { cat("derivatives() error:",   conditionMessage(e), "\n"); NULL },
    warning = function(w) { cat("derivatives() warning:", conditionMessage(w), "\n"); NULL }
  )
  if (!is.null(test_deriv)) cat("  OK — rows:", nrow(test_deriv), "\n")
}



#  Main loop
all_omnibus <- list()
all_sensitiv <- list()
all_curves <- list()
all_derivs <- list()
all_diag <- list()
failed <- c()

for (i in seq_along(triplets)) {
  asv_id <- names(triplets)[i]
  triplet <- triplets[[i]]
  cat(sprintf("[%02d/%02d] %s\n", i, n_triplets, asv_id))

  result <- tryCatch({
    df_long <- make_long(df, triplet)
    m_joint <- fit_joint_gam(df_long)
    m_constrained <- fit_constrained_gam(df_long)

    list(
      omnibus = extract_omnibus(m_joint, m_constrained, asv_id),
      sensitiv = extract_sensitivity(m_joint, asv_id),
      curves = extract_diff_smooth_curves(m_joint, asv_id),
      derivs = extract_derivative_curves(m_joint, asv_id),   # NEW
      diag = extract_diagnostics(m_joint, asv_id)
    )
  },
  error = function(e) {
    message(sprintf("  ERROR [%s]: %s\n  Call: %s", asv_id, conditionMessage(e),
                    deparse(conditionCall(e))))
    NULL
  },
  warning = function(w) {
    message(sprintf("  WARNING [%s]: %s", asv_id, conditionMessage(w)))
    invokeRestart("muffleWarning")
  })

  if (!is.null(result)) {
    all_omnibus[[i]] <- result$omnibus
    all_sensitiv[[i]] <- result$sensitiv
    all_curves[[i]] <- result$curves
    all_derivs[[i]] <- result$derivs
    all_diag[[i]] <- result$diag
  } else {
    failed <- c(failed, asv_id)
  }
}

cat(sprintf("\nCompleted: %d/%d  |  Failed: %d\n",
            n_triplets - length(failed), n_triplets, length(failed)))
if (length(failed) > 0)
  cat("Failed ASVs:", paste(failed, collapse = ", "), "\n")

omnibus_all <- bind_rows(all_omnibus)
sensitiv_all <- bind_rows(all_sensitiv)
curves_all <- bind_rows(all_curves)
derivs_all <- bind_rows(all_derivs)   # NEW
diff_summary <- summarise_diff_smooth(curves_all)
diag_all <- bind_rows(all_diag)


# BH-FDR correction
omnibus_all <- omnibus_all %>%
  mutate(p_adj_BH = p.adjust(p_raw, method = "BH"),
         sig_BH   = p_adj_BH < 0.05)
cat(sprintf("\n%d / %d ASV triplets significant (BH q < 0.05)\n",
            sum(omnibus_all$sig_BH, na.rm = TRUE), nrow(omnibus_all)))



# Aggregate sensitivity
sensitivity_summary <- sensitiv_all %>%
  group_by(predictor, dtype) %>%
  summarise(
    n_triplets = n(),
    grand_mean = mean(mean_abs_deriv, na.rm = TRUE),
    grand_se = sd(mean_abs_deriv,   na.rm = TRUE) / sqrt(n()),
    grand_sd = sd(mean_abs_deriv,   na.rm = TRUE),
    ci_lower = grand_mean - 1.96 * grand_se,
    ci_upper = grand_mean + 1.96 * grand_se,
    I2 = pmax(0, pmin(100,
      100 * (grand_sd^2 - mean(sd_abs_deriv^2 / n_points, na.rm = TRUE)) /
        grand_sd^2)),
    mean_prop_sig = mean(prop_sig_region, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(dtype_label = dtype_labels[dtype])

diff_consistency <- diff_summary %>%
  group_by(predictor, pair_label) %>%
  summarise(
    n_triplets = n(),
    pct_any_sig = 100 * mean(prop_sig > 0),
    mean_prop_sig = mean(prop_sig),
    mean_net_dir  = mean(net_direction),
    .groups = "drop"
  )


# Diagnostic summary
cat("\n=== Diagnostic summary ===\n")
print(diag_all[, c("asv_id","dev_expl_pct","max_concurvity",
                   "n_k_fail","n_near_zero_edf","any_flag")],
      row.names = FALSE)

n_flagged <- sum(diag_all$any_flag, na.rm = TRUE)
cat(sprintf("\n%d / %d models flagged (k-fail, concurvity > 0.8, or dev.expl < 20%%)\n",
            n_flagged, nrow(diag_all)))

if (n_flagged > 0) {
  cat("\nFlagged models:\n")
  flagged <- diag_all[diag_all$any_flag, ]
  for (j in seq_len(nrow(flagged))) {
    cat(sprintf("  %s: dev=%.1f%%  concurv=%.3f  k_fail=%d  near_zero=%d\n",
                flagged$asv_id[j], flagged$dev_expl_pct[j],
                flagged$max_concurvity[j], flagged$n_k_fail[j],
                flagged$n_near_zero_edf[j]))
    if (nchar(flagged$k_fail_terms[j])    > 0) cat("    k-fail:      ", flagged$k_fail_terms[j],    "\n")
    if (nchar(flagged$conc_high_terms[j]) > 0) cat("    concurvity:  ", flagged$conc_high_terms[j], "\n")
    if (nchar(flagged$near_zero_terms[j]) > 0) cat("    near-zero:   ", flagged$near_zero_terms[j], "\n")
  }
}



# Save all flat files
write.csv(omnibus_all, file.path(OUTPUT_DIR, "01_omnibus_tests.csv"), row.names = FALSE)
write.csv(sensitiv_all, file.path(OUTPUT_DIR, "02_sensitivity_per_triplet.csv"), row.names = FALSE)
write.csv(sensitivity_summary, file.path(OUTPUT_DIR, "03_sensitivity_summary.csv"), row.names = FALSE)
write.csv(curves_all, file.path(OUTPUT_DIR, "04_diff_smooth_curves.csv"), row.names = FALSE)
write.csv(diff_summary, file.path(OUTPUT_DIR, "05_diff_smooth_summary.csv"), row.names = FALSE)
write.csv(diff_consistency, file.path(OUTPUT_DIR, "06_diff_consistency.csv"), row.names = FALSE)
write.csv(diag_all, file.path(OUTPUT_DIR, "07_gam_diagnostics.csv"), row.names = FALSE)
write.csv(derivs_all, file.path(OUTPUT_DIR, "08_derivative_curves.csv"), row.names = FALSE)  # NEW

cat("\n--- Flat files saved to", OUTPUT_DIR, "---\n")
cat("  01–07: unchanged\n")
cat("  08_derivative_curves.csv — signed pointwise derivatives\n")
cat("         columns: asv_id, dtype, predictor, x_value, derivative, lower_ci, upper_ci\n")
cat("         used in Python for Option B: delta_{j,i}(x) = deriv_alt - deriv_rDNA\n")

sink(file.path(OUTPUT_DIR, "session_info.txt"))
cat("Run:", format(Sys.time()), "\n\n")
sessionInfo()
sink()
