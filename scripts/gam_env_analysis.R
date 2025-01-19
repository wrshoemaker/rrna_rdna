rm(list = ls())
getwd()
setwd("~/GitHub/rrna_rdna/")

library(mgcv)
library(data.table)
library(performance)


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

otu_afd_all <- grep('Otu', colnames(df_nonans_t), value=TRUE)

env_variables <- c('otu_afd', 'p_value_or_coeff',  'water_temp', 'specific_conductivity', 'salinity',  'total_nitrogen', 'total_phosphorus', 'doc', 'secchi_depth', 'ph', 'dissolved_oxygen')
#sd_env_var <- c('std_dev', sd(df_nonans_t$water_temp), sd(df_nonans_t$total_nitrogen), sd(df_nonans_t$total_phosphorus), sd(df_nonans_t$doc), sd(df_nonans_t$secchi_depth), sd(df_nonans_t$ph), sd(df_nonans_t$dissolved_oxygen))

# generate empty matrix
# (entry, nrows, ncolumns)
# p values and coeff for each AFD (twice as much)
matrix <- matrix(NA, (2*length(otu_afd_all))+1, length(env_variables))
matrix[1,] <- env_variables

for (i in 1:length(otu_afd_all)) { 
  
  otu_afd_i <- otu_afd_all[i]
  model_i <- as.formula(paste(otu_afd_i, " ~ water_temp + specific_conductivity + salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + ph + dissolved_oxygen"))
  gam_env_i <- gam(formula=model_i, data=df_nonans_t)
  
  coef_gam_i <- as.numeric(gam_env_i$coefficients[2:10])
  p_value_i <- as.numeric(summary(gam_env_i)$p.pv)[2:10]
  
  coef_gam_out_i <- c(otu_afd_i, 'coeff',  coef_gam_i)
  p_value_out_i <- c(otu_afd_i, 'p_value',  p_value_i)
  
  matrix[(2*i),] <- coef_gam_out_i
  matrix[(2*i)+1,] <- p_value_out_i
  
}



#save(copy_fourgram, file = "data.")

write.table(x = matrix, file = "data/gam_env_analysis.csv", sep = ',', row.names = FALSE, col.names = FALSE, quote=FALSE)





#capture.output(gam.check(gam_env_i))

#summary(gam_env_i)

#model <- as.formula("clr_afd_otu1_rna_dna ~ water_temp+total_nitrogen+total_phosphorus+doc + secchi_depth + ph + dissolved_oxygen")
#gam_env <- gam(formula=model, data=df_nonans_t)

# check residuals 
#gam.check(gam_env)


# check colinearity
#check_collinearity(gam_env)
#plot(check_collinearity(gam_env))

# colinearity is low (largest VAF = 3.20), continue with analysis
#summary(gam_env)

# DOC is only significant coefficient


#sd_env_var <- c(sd(df_nonans_t$water_temp), sd(df_nonans_t$total_nitrogen), sd(df_nonans_t$total_phosphorus), sd(df_nonans_t$doc))

#coef_test_gam <- gam_env$coefficients[2:5]*sd_env_var


#coef_test_gam

