#GAM Sensitivity Comparison — 21 ASV Triplets (_dna / _rna / _rna_dna)

# Pipeline
# 1. Read & transpose data → samples × variables
# 2. Detect 21 triplets from _dna / _rna / _rna_dna suffixes
# 3. Fit joint GAM per triplet (response × smooth interaction)
# 4. Extract omnibus tests, sensitivity indices, difference smooth curves
# 5. Apply BH-FDR correction jointly across all 693 tests
# 6. Save ALL figure-ready flat files

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

#DATA_PATH <- read.csv('data/data_for_gam.csv', sep=',')
DATA_PATH     <- "data/data_for_gam.csv"
OUTPUT_DIR    <- "data/gam_results"
#tree <- read.tree("asvs.nwk")
dir.create(OUTPUT_DIR, showWarnings = FALSE)


env_predictors  <- c(
  "water_temp", "specific_conductivity", "salinity",
  "total_nitrogen", "total_phosphorus", "doc",
  "secchi_depth", "ph", "dissolved_oxygen"
)


temp_predictors <- c("day_of_year", "days")
all_predictors  <- c(env_predictors, temp_predictors)

dtype_labels <- c(
  dna = "DNA",
  rna = "RNA",
  rna_dna = "RNA:DNA"
)

raw <- read.csv(DATA_PATH, check.names = FALSE)
# Transpose: samples become rows, variables become columns
row_names <- raw$sample_number
# numeric matrix: 75 vars × 123 samples
mat <- as.matrix(raw[, -1])
# 123 samples × 75 vars
df <- as.data.frame(t(mat))
colnames(df) <- row_names
df <- df %>% mutate(across(everything(), as.numeric))

#stopifnot(all(env_predictors %in% colnames(df)))
#stopifnot(all(temp_predictors %in% colnames(df)))

all_cols  <- colnames(df)
asv_cols  <- all_cols[grepl("^ASV_", all_cols)]

# get ASV name
get_base <- function(x) {
  x <- sub("_rna_dna$", "", x)
  x <- sub("_rna$", "", x)
  x <- sub("_dna$", "", x)
  x
}

get_dtype <- function(x) {ifelse(grepl("_rna_dna$", x), "rna_dna", ifelse(grepl("_rna$", x), "rna", "dna"))}

asv_df <- data.frame(
  full_name = asv_cols,
  base_name = get_base(asv_cols),
  dtype     = get_dtype(asv_cols),
  stringsAsFactors = FALSE
)

base_names  <- unique(asv_df$base_name)
n_triplets  <- length(base_names)
asv_ids     <- sprintf("ASV_%03d", seq_len(n_triplets))
id_map      <- setNames(asv_ids, base_names)   # base_name → short ID
asv_df$asv_id    <- id_map[asv_df$base_name]
asv_df$short_col <- paste(asv_df$asv_id, asv_df$dtype, sep = "_")

write.csv(asv_df, file.path(OUTPUT_DIR, "asv_id_lookup.csv"), row.names = FALSE)

# Build list of triplets: each element = named vector (dna, rna, rna_dna=> full col name)
triplets <- lapply(base_names, function(base) {
  sub <- asv_df[asv_df$base_name == base, ]
  setNames(sub$full_name, sub$dtype)   # names: 'dna', 'rna', 'rna_dna'
})
names(triplets) <- asv_ids

incomplete <- sapply(triplets, length) != 3
if (any(incomplete)) {
  warning("Incomplete triplets: ", paste(names(which(incomplete)), collapse = ", "))
  triplets <- triplets[!incomplete]
}

# Helper functions
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
    data   = df_long,
    method = "REML"
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
    data   = df_long,
    method = "REML"
  )
}

# Omnibus LRT: joint vs constrained model
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


# Scalar sensitivity: mean |f'| per predictor × dtype
extract_sensitivity <- function(m_joint, asv_id) {
  results      <- list()
  smooth_names <- gratia::smooths(m_joint)
  
  for (pred in all_predictors) {
    for (dt in c("dna", "rna", "rna_dna")) {
      candidate <- paste0("s(", pred, "):dtype", dt)
      
      if (!any(grepl(candidate, smooth_names, fixed = TRUE))) {
        message(sprintf("  Term not found: %s [skipping]", candidate))
        next
      }
      
      d <- tryCatch(
        gratia::derivatives(
          m_joint,
          term = candidate,
          type = "central",
          interval = "simultaneous",
          unconditional = TRUE
        ),
        error   = function(e) {
          message(sprintf("  derivatives() error %s|%s|%s: %s",
                          asv_id, pred, dt, conditionMessage(e)))
          NULL
        },
        warning = function(w) {
          message(sprintf("  derivatives() warning %s|%s|%s: %s",
                          asv_id, pred, dt, conditionMessage(w)))
          NULL
        }
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




# Pairwise difference smooth curves — save FULL pointwise data for plotting
extract_diff_smooth_curves <- function(m_joint, asv_id) {
  pairs <- list(
    c("dna", "rna"),
    c("dna", "rna_dna"),
    c("rna", "rna_dna")
  )
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
        gratia::difference_smooths(
          m_joint,
          smooth        = smooth_term,
          levels        = pair,
          ci_level      = 0.95,
          unconditional = TRUE
        ),
        error = function(e) {
          message(sprintf("  diff_smooth error %s|%s|%s vs %s: %s",
                          asv_id, pred, pair[1], pair[2], conditionMessage(e)))
          NULL
        }
      )
      
      if (is.null(d) || nrow(d) == 0) next
      
      # difference_smooths() returns ALL pairwise combinations regardless of
      # the levels argument — filter to the specific pair before extracting
      d <- d[d$.level_1 == pair[1] & d$.level_2 == pair[2], ]
      
      if (nrow(d) == 0) next
      
      col_diff  <- find_col(d, c(".diff",     "difference",  ".difference"))
      col_lower <- find_col(d, c(".lower_ci", "lower"))
      col_upper <- find_col(d, c(".upper_ci", "upper"))
      
      if (any(is.na(c(col_diff, col_lower, col_upper)))) {
        message(sprintf(
          "  Cannot map diff/lower/upper for %s|%s. Columns: %s",
          asv_id, pred, paste(colnames(d), collapse = ", ")
        ))
        next
      }
      
      known_non_x <- c(col_diff, col_lower, col_upper,
                       ".smooth", ".by", ".level_1", ".level_2", ".se")
      x_col <- intersect(all_predictors, colnames(d))
      if (length(x_col) == 0) {
        leftover <- setdiff(colnames(d), known_non_x)
        x_col    <- leftover[sapply(d[leftover], is.numeric)][1]
      }
      
      if (length(x_col) == 0 || is.na(x_col)) {
        message(sprintf("  Cannot identify x column for %s|%s", asv_id, pred))
        next
      }
      
      out <- d %>%
        rename(
          x_value    = all_of(x_col[[1]]),
          difference = all_of(col_diff),
          lower      = all_of(col_lower),
          upper      = all_of(col_upper)
        ) %>%
        mutate(
          asv_id     = asv_id,
          predictor  = pred,
          dtype_a    = pair[1],
          dtype_b    = pair[2],
          pair_label = paste(dtype_labels[pair[1]], "vs",
                             dtype_labels[pair[2]])
        ) %>%
        select(asv_id, predictor, dtype_a, dtype_b, pair_label,
               x_value, difference, lower, upper)
      
      results[[length(results) + 1]] <- out
    }
  }
  bind_rows(results)
}



# Pairwise summary stats (for consistency heatmap => one row per predictor × pair)
summarise_diff_smooth <- function(curve_data) {
  curve_data %>%
    group_by(asv_id, predictor, dtype_a, dtype_b, pair_label) %>%
    summarise(
      prop_sig = mean((lower > 0) | (upper < 0)),
      net_direction = mean(difference),
      max_abs_diff  = max(abs(difference)),
      .groups = "drop"
    )
}




asv_check <- grep("^ASV_", colnames(df), value = TRUE)
cat("ASV columns found in df:", length(asv_check), "\n")
cat("First ASV column name:\n ", asv_check[1], "\n")



# named character — WRONG for .data[[]]
cat("\nTriplet indexing test:\n")
cat("  Single bracket triplets[[1]]['dna']  class:", 
    class(triplets[[1]]["dna"]),
    "\n")

# plain character — CORRECT
cat("  Double bracket triplets[[1]][['dna']] class:", 
    class(triplets[[1]][["dna"]]),
    "\n")

cat("\nTesting make_long on ASV_001...\n")
test_long <- tryCatch({
  make_long(df, triplets[[1]])
}, error = function(e) {
  cat("make_long error:", conditionMessage(e), "\n")
  NULL
})
if (!is.null(test_long)) {
  cat("  OK — rows:", nrow(test_long), "| dtype levels:", 
      levels(test_long$dtype), "\n")
}




cat("\nTesting fit_joint_gam on ASV_001...\n")
test_gam <- tryCatch({
  fit_joint_gam(test_long)
}, error   = function(e) { cat("GAM error:",   conditionMessage(e), "\n"); NULL },
warning = function(w) { cat("GAM warning:", conditionMessage(w), "\n"); NULL })
if (!is.null(test_gam)) cat("  OK — deviance explained:", 
                            round(summary(test_gam)$dev.expl * 100, 1), "%\n")


test_diff <- gratia::difference_smooths(
  test_gam,
  smooth = "s(water_temp)",
  levels = c("dna", "rna"),
  ci_level = 0.95
)

cat("Columns returned by difference_smooths():\n")
print(colnames(test_diff))
cat("\nHead:\n")
print(head(test_diff, 3))




# Test derivatives on triplet 1 if GAM fitted
if (!is.null(test_gam)) {
  cat("\nTesting derivatives() on ASV_001 / water_temp / dna...\n")
  test_deriv <- tryCatch(
    gratia::derivatives(test_gam, term = "s(water_temp):dtypedna", type = "central", interval = "simultaneous", unconditional = TRUE),
    error   = function(e) { cat("derivatives() error:",   conditionMessage(e), "\n"); NULL },
    warning = function(w) { cat("derivatives() warning:", conditionMessage(w), "\n"); NULL }
  )
  if (!is.null(test_deriv)) cat("  OK — rows:", nrow(test_deriv), "\n")
}



# Main loop
all_omnibus <- list()
all_sensitiv <- list()
all_curves <- list()
failed <- c()

for (i in seq_along(triplets)) {
  asv_id  <- names(triplets)[i]
  triplet <- triplets[[i]]
  cat(sprintf("[%02d/%02d] %s\n", i, n_triplets, asv_id))
  
  result <- tryCatch({
    df_long <- make_long(df, triplet)
    m_joint <- fit_joint_gam(df_long)
    m_constrained <- fit_constrained_gam(df_long)
    
    list(
      omnibus = extract_omnibus(m_joint, m_constrained, asv_id),
      sensitiv = extract_sensitivity(m_joint, asv_id),
      curves = extract_diff_smooth_curves(m_joint, asv_id)
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
  } else {
    failed <- c(failed, asv_id)
  }
}

cat(sprintf("\nCompleted: %d/%d  |  Failed: %d\n", n_triplets - length(failed), n_triplets, length(failed)))
if (length(failed) > 0)
  cat("Failed ASVs:", paste(failed, collapse = ", "), "\n")

omnibus_all  <- bind_rows(all_omnibus)
sensitiv_all <- bind_rows(all_sensitiv)
curves_all   <- bind_rows(all_curves)
diff_summary <- summarise_diff_smooth(curves_all)


# 5  BH-FDR CORRECTION

omnibus_all <- omnibus_all %>% mutate(p_adj_BH = p.adjust(p_raw, method = "BH"), sig_BH   = p_adj_BH < 0.05)
cat(sprintf("\n%d / %d ASV triplets significant (BH q < 0.05)\n", sum(omnibus_all$sig_BH, na.rm = TRUE), nrow(omnibus_all)))


# test <- difference_smooths(m, smooth = "s(water_temp)", 
#                           levels = c("dna", "rna_dna"))
# if water_temp is warmer = rna_dna more sensitive,
# the difference (dna - rna_dna)  NEGATIVE at high temps


#######
# test sign
#df_long_test <- make_long(df, triplets[["ASV_001"]])
#m_test <- fit_joint_gam(df_long_test)
# Get difference smooth: dna - rna_dna
# High water temp, rRNA:rDNA should be MORE sensitive than rDNA
# so (dna - rna_dna) should be NEGATIVE at high temps
#test_diff <- difference_smooths(m_test, smooth = "s(water_temp)", levels = c("dna", "rna_dna"))
# Check sign at low vs high water temp
# low temp - expect positive or near zero
#head(test_diff[order(test_diff$water_temp), ], 3)
# high temp - expect negative
#tail(test_diff[order(test_diff$water_temp), ], 3)
# CORRECT!
#######


# 6 Aggregate sensitivity across triplets
sensitivity_summary <- sensitiv_all %>%
  group_by(predictor, dtype) %>%
  summarise(
    n_triplets = n(),
    grand_mean = mean(mean_abs_deriv, na.rm = TRUE),
    grand_se = sd(mean_abs_deriv, na.rm = TRUE) / sqrt(n()),
    grand_sd = sd(mean_abs_deriv, na.rm = TRUE),
    ci_lower = grand_mean - 1.96 * grand_se,
    ci_upper = grand_mean + 1.96 * grand_se,
    I2 = pmax(0, pmin(100, 100 * (grand_sd^2 - mean(sd_abs_deriv^2 / n_points, na.rm = TRUE)) / grand_sd^2)),
    mean_prop_sig = mean(prop_sig_region, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(dtype_label = dtype_labels[dtype])

diff_consistency <- diff_summary %>%
  group_by(predictor, pair_label) %>%
  summarise(
    n_triplets    = n(),
    pct_any_sig   = 100 * mean(prop_sig > 0),
    mean_prop_sig = mean(prop_sig),
    mean_net_dir  = mean(net_direction),
    .groups       = "drop"
  )



write.csv(omnibus_all, file.path(OUTPUT_DIR, "01_omnibus_tests.csv"), row.names = FALSE)
write.csv(sensitiv_all, file.path(OUTPUT_DIR, "02_sensitivity_per_triplet.csv"), row.names = FALSE)
write.csv(sensitivity_summary, file.path(OUTPUT_DIR, "03_sensitivity_summary.csv"), row.names = FALSE)
write.csv(curves_all, file.path(OUTPUT_DIR, "04_diff_smooth_curves.csv"), row.names = FALSE)
write.csv(diff_summary, file.path(OUTPUT_DIR, "05_diff_smooth_summary.csv"), row.names = FALSE)
write.csv(diff_consistency,file.path(OUTPUT_DIR, "06_diff_consistency.csv"), row.names = FALSE)
cat("\n--- Flat files saved to", OUTPUT_DIR, "---\n")

