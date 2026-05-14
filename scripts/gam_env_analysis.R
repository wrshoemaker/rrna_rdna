rm(list = ls())
rm()
getwd()
setwd("~/GitHub/rrna_rdna/")

detach("package:CausalGAM", unload=TRUE)
detach("package:gam", unload=TRUE)

library(mgcv)
library(data.table)
library(performance)
library(moments)


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

otu_afd_all <- grep('ASV', colnames(df_nonans_t), value=TRUE)

env_variables <- c('otu_afd', 'p_value_or_coeff',  'water_temp', 'specific_conductivity', 'salinity',  'total_nitrogen', 'total_phosphorus', 'doc', 'secchi_depth', 'ph', 'dissolved_oxygen')
#sd_env_var <- c('std_dev', sd(df_nonans_t$water_temp), sd(df_nonans_t$total_nitrogen), sd(df_nonans_t$total_phosphorus), sd(df_nonans_t$doc), sd(df_nonans_t$secchi_depth), sd(df_nonans_t$ph), sd(df_nonans_t$dissolved_oxygen))

# generate empty matrix
# (entry, nrows, ncolumns)
# p values and coeff for each AFD (twice as much)
matrix <- matrix(NA, (2*length(otu_afd_all))+1, length(env_variables))
matrix[1,] <- env_variables

for (i in 1:length(otu_afd_all)) { 
  
  otu_afd_i <- otu_afd_all[i]
  #model_i <- as.formula(paste(otu_afd_i, " ~ water_temp + specific_conductivity + salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + ph + dissolved_oxygen"))
  #model_i <- as.formula(paste(otu_afd_i, " ~ water_temp + specific_conductivity + salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + ph + dissolved_oxygen + s(days)"))
  #model_i <- as.formula(paste(otu_afd_i, " ~ water_temp + specific_conductivity + salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + ph + dissolved_oxygen + s(day_of_year,k=", 12, ",  bs='cc') + s(days, k=10, bs='tp')") )

  model_i <- as.formula(paste(otu_afd_i, " ~ s(water_temp,k=5) + s(specific_conductivity,k=5) + s(salinity,k=5) + s(total_nitrogen,k=5) + s(total_phosphorus,k=5) + s(doc,k=5) + s(secchi_depth,k=5) + s(ph,k=5) + s(dissolved_oxygen,k=5) + s(day_of_year,k=", 12, ",  bs='cc') + s(days, k=10, bs='tp')") )
  #model_i <- as.formula(paste(otu_afd_i, " ~ s(water_temp,k=5) + s(specific_conductivity,k=5) + s(salinity,k=5) + s(total_nitrogen,k=5) + s(total_phosphorus,k=5) + s(doc,k=5) + s(secchi_depth,k=5) + s(ph,k=5) + s(dissolved_oxygen,k=5) + s(days, k=10, bs='tp')") )
  
  # represents degrees of freedom
  # 123 samples
  #model_i <- as.formula(paste(otu_afd_i, " ~ water_temp + specific_conductivity + salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + ph + dissolved_oxygen + s(days,k=", 8, ") + s(day_of_year, bs = 'cc')"))
    
  #sk <- skewness(df[[otu_i]])
  #family_i <- if (abs(sk) < 1) gaussian() else scat()
  
  
  gam_env_i <- gam(formula=model_i, data=df_nonans_t, family=scat(),  method  = "REML",  select  = TRUE, knots= list(day_of_year = c(1, 365)))
   
  term_order <- c(
    "s(water_temp)", "s(specific_conductivity)", "s(salinity)",
    "s(total_nitrogen)", "s(total_phosphorus)", "s(doc)",
    "s(secchi_depth)", "s(ph)", "s(dissolved_oxygen)")
  
  s_table <- summary(gam_env_i)$s.table
  coef_gam_i  <- s_table[term_order, "edf"]
  p_value_i   <- s_table[term_order, "p-value"]
  
  
  #coef_gam_i <- as.numeric(gam_env_i$coefficients[2:10])
  #p_value_i <- as.numeric(summary(gam_env_i)$p.pv)[2:10]
  #p_value_i <- as.numeric(summary(gam_env_i)$p.pv)[2:10]
  #p_value_i <- summary(gam_env_i)$s.table[, "p-value"]
  
  concurvity(gam_env_i, full = FALSE)$worst
  
  
  
  coef_gam_out_i <- c(otu_afd_i, 'coeff',  coef_gam_i)
  p_value_out_i <- c(otu_afd_i, 'p_value',  p_value_i)
  
  matrix[(2*i),] <- coef_gam_out_i
  matrix[(2*i)+1,] <- p_value_out_i
  
  #print(otu_afd_i)
  #gam.check(gam_env_i)
  
}



#save(copy_fourgram, file = "data.")
write.table(x = matrix, file = "data/gam_env_analysis.csv", sep = ',', row.names = FALSE, col.names = FALSE, quote=FALSE)




# Otu000002, Otu000003, Otu000019, Otu000028, Otu000032


# test OTU1
#model_otu1_dna <- as.formula("Otu000004_dna ~ water_temp + specific_conductivity + salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + ph + dissolved_oxygen + s(days,k=8) + s(day_of_year, bs = 'cc')")
#model_otu1_dna <- as.formula("Otu000001_rna ~ water_temp + specific_conductivity + salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + ph + dissolved_oxygen +  mgcv::s(days,k=20)")
model_otu1_dna <- as.formula("Otu000001_rna ~ water_temp + specific_conductivity + 
    salinity + total_nitrogen + total_phosphorus + doc + secchi_depth + 
    ph + dissolved_oxygen + s(days, k=20)")

gam_env_otu1_dna <- gam(formula=model_otu1_dna, data=df_nonans_t)

#k.check(gam_env_otu1_dna, subsample=5000, n.rep=400)
gam.check(gam_env_otu1_dna)
capture.output(gam.check(gam_env_otu1_dna))

summary(gam_env_otu1_dna)

concurvity(gam_env_otu1_dna, full = FALSE)


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

