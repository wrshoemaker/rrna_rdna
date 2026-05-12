rm(list = ls())
getwd()
setwd("~/GitHub/rrna_rdna/")

install.packages("gam")
packageurl <- "https://cran.r-project.org/src/contrib/Archive/CausalGAM/CausalGAM_0.1-4.tar.gz"
install.packages(packageurl, repos = NULL, type = "source")

library(mgcv)
library(data.table)
library(performance)
library(CausalGAM)


set.seed(123456789)

df <- read.csv('data/data_for_gam.csv', sep=',')

# remove NaNs, transpose, set column labels
df_nonans <- df[,colSums(is.na(df)) < nrow(df)]
df_nonans <- df[,colSums(is.na(df)) == 0]


df_nonans_t <- transpose(df_nonans)
colnames(df_nonans_t) <- unlist(df_nonans_t[1, ])
df_nonans_t <- tail(df_nonans_t, -1)

# make sure data is numeric
df_nonans_t <- data.frame(apply(df_nonans_t, 2, function(x) as.numeric(as.character(x))))

colnames(df_nonans_t)

# binarize the data using the median
df_nonans_t$water_temp_high  <- as.integer(df_nonans_t$water_temp >= median(df_nonans_t$water_temp))
df_nonans_t$Otu000001_dna_high <- as.integer(df_nonans_t$Otu000001_dna  >= median(df_nonans_t$Otu000001_dna))

# Temperature -> phototroph

cols_A <- c("water_temp_high", "water_temp", "Otu000001_dna", "days")
df_A   <- df_nonans_t[complete.cases(df_nonans_t[, cols_A]), cols_A]

ATE_A <- estimate.ATE(
  pscore.formula    = water_temp_high ~ s(days),
  pscore.family     = binomial,
  outcome.formula.t = Otu000001_rna ~ s(days),             # continuous
  outcome.formula.c = Otu000001_rna ~ s(days),
  outcome.family    = gaussian,                            # not binomial
  treatment.var     = "water_temp_high",
  data              = df_A,
  divby0.action     = "truncate",
  divby0.tol        = 0.001,
  var.gam.plot      = FALSE,
  nboot             = 200,
  suppress.warnings = TRUE
)

print(ATE_A)



cols_B <- c("water_temp_high", "water_temp","Otu000001_dna",  "Otu000001_dna_high", "days")
df_B  <- df_nonans_t[complete.cases(df_nonans_t[, cols_B]), cols_B]


ATE_B <- tryCatch(
  estimate.ATE(
    pscore.formula    = Otu000001_dna_high ~ s(water_temp) + s(days),
    pscore.family     = binomial,
    outcome.formula.t = Otu000002_dna ~ s(water_temp) + s(days),
    outcome.formula.c = Otu000002_dna ~ s(water_temp) + s(days),
    outcome.family    = gaussian,
    treatment.var     = "Otu000001_dna_high",
    data              = df_nonans_t,
    divby0.action     = "truncate",
    divby0.tol        = 0.001,
    var.gam.plot      = FALSE,
    nboot             = 200,
    suppress.warnings = TRUE
  ),
  error = function(e) { message("ERROR: ", e$message); NULL }
)

print(ATE_B)



#### 

fit_temp  <- mgcv::gam(water_temp     ~ s(days, bs = "cc"), data = df_B)
fit_photo <- mgcv::gam(Otu000001_dna  ~ s(days, bs = "cc"), data = df_B)
fit_hetero <- mgcv::gam(Otu000001_dna  ~ s(days, bs = "cc"), data = df_B)

df_B$temp_anomaly  <- residuals(fit_temp)
df_B$photo_anomaly <- residuals(fit_photo)
df_B$hetero_anomaly <- residuals(fit_hetero)

df_B$temp_anom_high  <- as.integer(df_B$temp_anomaly  >= 0)
df_B$photo_anom_high <- as.integer(df_B$photo_anomaly >= 0)

lag <- 20
#df_B$temp_anom_high_lag <- c(rep(NA, lag), head(df_B$temp_anom_high, -lag))
df_B$temp_anom_high_lag <- c(rep(NA, lag), head(df_B$temp_anom_high, -lag))

df_B_test <- na.omit(df_B)


ATE_A <- estimate.ATE(
  pscore.formula    = temp_anom_high_lag ~ s(days),
  pscore.family     = binomial,
  outcome.formula.t = photo_anomaly ~ s(days),
  outcome.formula.c = photo_anomaly ~ s(days),
  outcome.family    = gaussian,
  treatment.var     = "temp_anom_high_lag",   # must match pscore LHS
  data              = df_B_test,
  divby0.action     = "truncate",
  divby0.tol        = 0.001,
  var.gam.plot      = FALSE,
  nboot             = 200,
  suppress.warnings = TRUE
)


print(ATE_A)




ATE_B <- tryCatch(
  estimate.ATE(
    pscore.formula    = photo_anom_high ~ s(temp_anomaly) + s(days),
    pscore.family     = binomial,
    outcome.formula.t = hetero_anomaly ~ s(temp_anomaly) + s(days),
    outcome.formula.c = hetero_anomaly ~ s(temp_anomaly) + s(days),
    outcome.family    = gaussian,
    treatment.var     = "photo_anom_high",
    data              = df_B,
    divby0.action     = "truncate",
    divby0.tol        = 0.001,
    var.gam.plot      = FALSE,
    nboot             = 200,
    suppress.warnings = TRUE
  ),
  error = function(e) { message("ERROR for ", e$message); NULL }
)


print(ATE_B)


