# Load necessary libraries
library(rrBLUP)
library(caret)  # For creating folds
library(dplyr)
setwd("/Users/thulanihewavithana/Documents/PhD/Courses/CMPT898/CMPT-PLSC_819_Project/Dataset4/")
SW <- read.csv("BR.csv") 
SW$SW <- as.numeric(SW$Blast_resistance)

# Calculate average SW by Name
averaged_data <- SW %>%
  group_by(Name) %>%
  summarise(Average_SW = mean(SW, na.rm = TRUE))

write.table(averaged_data, "averaged_SW.txt", sep = "\t", row.names = FALSE, quote = FALSE)


# View the result
print(averaged_data)

memory.limit(size = 16000)  # Set limit to 16 GB

library(data.table)



X <- as.matrix(fread("genotype_matrix.csv"))

# Rows to be removed
rows_to_remove <- c(13, 29, 50, 78, 97, 99, 126, 133, 154, 186, 199, 200, 201, 237, 261, 276, 280, 309, 321, 347, 356, 363)

# Remove the rows from the matrix X
X <- X[-rows_to_remove, ]


y <- as.numeric(read.table("averaged_SW.txt", sep = '\t', header = T)$Average_SW) # Phenotype vector
# y <- y[2:196]


# Check for missing data and impute if necessary
X[is.na(X)] <- 0  # Replace missing SNPs with 0
y[is.na(y)] <- mean(y, na.rm = TRUE)  # Impute missing phenotypes with mean

# Ensure row alignment between genotype (X) and phenotype (y)
stopifnot(nrow(X) == length(y))  # Ensure consistency



############RRBLUP

# Compute the Genomic Relationship Matrix (GRM)
GRM <- A.mat(X)  # Using rrBLUP to calculate the GRM

# Fit the RR-BLUP Model
model <- mixed.solve(y = y, K = GRM)

# Extract GEBVs (Genomic Estimated Breeding Values)
GEBVs <- model$u  # Breeding values

# Predicted Phenotypes (GEBVs)
# Convert GEBVs to a numeric vector
GEBVs <- as.numeric(model$u)  # Ensure GEBVs is a numeric vector

# Add the intercept (model$beta) to each GEBV to calculate predicted phenotypes
predicted_phenotypes <- GEBVs + model$beta  # Add scalar intercept to each element in GEBVs

# Observed phenotypes
observed_phenotypes <- y

# Calculate R²
ss_total <- sum((observed_phenotypes - mean(observed_phenotypes))^2)  # Total sum of squares
ss_residual <- sum((observed_phenotypes - predicted_phenotypes)^2)    # Residual sum of squares
r2 <- 1 - (ss_residual / ss_total)                                   # R² formula

# Display results
cat("R²:", r2, "\n")

#####################GBLUP##########
# Compute the Genomic Relationship Matrix (G matrix)
G_matrix <- Gmatrix(X, method = "VanRaden")  # Using `Gmatrix` from the `pedigree` package

# Prepare `ped` data
ped <- data.frame(
  ID = 1:length(y),
  P = y
)

# Prepare marker effects matrix `M` (equivalent to the G matrix in your case)
rownames(X) <- ped$ID  # Set rownames for genotype matrix
M <- X  # Use genotype matrix as marker effects

# Set heritability (h2) and compute lambda
h2 <- 0.02
lambda <- 1 / h2 - 1

# Fit GBLUP model
sol <- gblup(P ~ 1, data = ped, M = G_matrix, lambda = lambda)

# Display results
print(head(ped))
cat("Model Summary:\n")
print(summary(sol))

# Extract breeding values (GEBVs)
GEBVs <- as.numeric(sol[, 1])  # Extract the first column as numeric breeding values

predicted_phenotypes <- GEBVs
observed_phenotypes <- ped$P

ss_total <- sum((observed_phenotypes - mean(observed_phenotypes))^2)  # Total sum of squares
ss_residual <- sum((observed_phenotypes - predicted_phenotypes)^2)    # Residual sum of squares
r2 <- 1 - (ss_residual / ss_total)                                   # R² formula
cat("R²:", r2, "\n")

####################

# Load necessary library
library(BGLR)
library(caret)  # For creating folds

# Define parameters for Bayesian Ridge Regression
nIter <- 10000  # Number of iterations for MCMC
burnIn <- 2000  # Burn-in period 

# Load genotype and phenotype data
# X <- as.matrix(read.csv("genotype_matrix.csv"))  # SNP matrix
y <- as.numeric(read.table("averaged_SW.txt", sep = '\t', header = T)$Average_SW)  # Phenotype vector
# y <- y[2:196]

# Check for missing data and impute if necessary
X[is.na(X)] <- 0  # Replace missing SNPs with 0
y[is.na(y)] <- mean(y, na.rm = TRUE)  # Impute missing phenotypes with mean

# Ensure row alignment between genotype (X) and phenotype (y)
stopifnot(nrow(X) == length(y))  # Ensure consistency

# Full Model (Optional: without CV, just for comparison)
brr_model <- BGLR(y = y, ETA = list(list(X = X, model = "BRR")), nIter = nIter, burnIn = burnIn)

# Extract predicted values (Genomic Estimated Breeding Values - GEBVs)
GEBVs <- brr_model$yHat

# Plot observed vs. predicted for the full model
plot(y, GEBVs, main = "Observed vs Predicted (BRR)", xlab = "Observed", ylab = "Predicted")
abline(0, 1, col = "red")

# 7-Fold Cross-Validation Setup
set.seed(123)  # For reproducibility
folds <- createFolds(y, k = 7)

# Initialize list to store CV results and aggregated predictions
cv_results_bayes <- list()
predicted_all <- numeric(length(y))  # Store all test set predictions
observed_all <- numeric(length(y))  # Store all test set observations

# Perform 7-fold CV
for (i in seq_along(folds)) {
  cat("\nProcessing Fold:", i, "\n")
  
  # Training and test indices
  test_indices <- folds[[i]]
  train_indices <- setdiff(1:length(y), test_indices)
  
  # Training and test data
  y_train <- y[train_indices]
  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  
  # Fit the Bayesian Ridge Regression model
  brr_model <- BGLR(
    y = y_train,
    ETA = list(list(X = X_train, model = "BRR")),
    nIter = nIter,
    burnIn = burnIn
  )
  
  # Predict on test set
  GEBVs_test <- X_test %*% brr_model$ETA[[1]]$b + brr_model$mu
  
  # Store test set predictions and observations
  predicted_all[test_indices] <- GEBVs_test
  observed_all[test_indices] <- y[test_indices]
  
  # Calculate prediction accuracy (correlation)
  cor_test <- cor(y[test_indices], GEBVs_test)
  
  # Calculate R^2 (coefficient of determination)
  ss_total <- sum((y[test_indices] - mean(y[test_indices]))^2)
  ss_residual <- sum((y[test_indices] - GEBVs_test)^2)
  r2_test <- 1 - (ss_residual / ss_total)
  
  # Store results
  cv_results_bayes[[i]] <- list(
    fold = i,
    test_indices = test_indices,
    predicted = GEBVs_test,
    observed = y[test_indices],
    correlation = cor_test,
    r2 = r2_test
  )
  
  cat("Correlation for Fold", i, ":", cor_test, "\n")
  cat("R^2 for Fold", i, ":", r2_test, "\n")
}

# Calculate average correlation and R^2 across folds
correlations_bayes <- sapply(cv_results_bayes, function(res) res$correlation)
r2_values_bayes <- sapply(cv_results_bayes, function(res) res$r2)

mean_correlation_bayes <- mean(correlations_bayes)
mean_r2_bayes <- mean(r2_values_bayes)

# Calculate global test set R^2
global_ss_total <- sum((observed_all - mean(observed_all))^2)
global_ss_residual <- sum((observed_all - predicted_all)^2)
global_r2 <- 1 - (global_ss_residual / global_ss_total)

cat("\nMean Correlation Across 7 Folds (Bayesian):", mean_correlation_bayes, "\n")
cat("Mean R^2 Across 7 Folds (Bayesian):", mean_r2_bayes, "\n")
cat("Global R^2 Across All Test Sets (Bayesian):", global_r2, "\n")

# Optional: Save results to file
write.csv(data.frame(
  Fold = seq_along(correlations_bayes),
  Correlation = correlations_bayes,
  R2 = r2_values_bayes
), "bayesian_cv_results.csv", row.names = FALSE)




# Load necessary libraries
library(BGLR)
library(caret)  # For creating folds

# Load genotype and phenotype data
# X <- as.matrix(read.csv("genotype_matrix_normalized.csv"))  # SNP matrix
y <- as.numeric(read.table("averaged_SW.txt", sep = '\t', header = T)$Average_SW)  # Phenotype vector

# Check for missing data and impute if necessary
X[is.na(X)] <- 0  # Replace missing SNPs with 0
y[is.na(y)] <- mean(y, na.rm = TRUE)  # Impute missing phenotypes with mean

# Define parameters for BayesB
nIter <- 10000  # Number of iterations for MCMC
burnIn <- 2000  # Burn-in period

# 7-Fold Cross-Validation Setup
set.seed(123)  # For reproducibility
folds <- createFolds(y, k = 7)

# Initialize list to store CV results and aggregated predictions
cv_results_bayesb <- list()
predicted_all <- numeric(length(y))  # Store all test set predictions
observed_all <- numeric(length(y))  # Store all test set observations

# RMSE Function
rmse <- function(observed, predicted) {
  sqrt(mean((observed - predicted)^2))
}

# Perform 7-fold CV
for (i in seq_along(folds)) {
  cat("\nProcessing Fold:", i, "\n")
  
  # Training and test indices
  test_indices <- folds[[i]]
  train_indices <- setdiff(1:length(y), test_indices)
  
  # Training and test data
  y_train <- y[train_indices]
  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  
  # Fit BayesB model
  bayesb_model <- BGLR(
    y = y_train,
    ETA = list(list(X = X_train, model = "BayesB")),
    nIter = nIter,
    burnIn = burnIn
  )
  
  # Predict GEBVs for test set
  GEBVs_test <- X_test %*% bayesb_model$ETA[[1]]$b + bayesb_model$mu
  
  # Store test set predictions and observations
  predicted_all[test_indices] <- GEBVs_test
  observed_all[test_indices] <- y[test_indices]
  
  # Calculate performance metrics
  cor_test <- cor(y[test_indices], GEBVs_test)
  ss_total <- sum((y[test_indices] - mean(y[test_indices]))^2)
  ss_residual <- sum((y[test_indices] - GEBVs_test)^2)
  r2_test <- 1 - (ss_residual / ss_total)
  rmse_test <- rmse(y[test_indices], GEBVs_test)
  
  # Store results
  cv_results_bayesb[[i]] <- list(
    fold = i,
    predicted = GEBVs_test,
    observed = y[test_indices],
    correlation = cor_test,
    r2 = r2_test,
    rmse = rmse_test
  )
  
  cat("Correlation for Fold", i, ":", cor_test, "\n")
  cat("R² for Fold", i, ":", r2_test, "\n")
  cat("RMSE for Fold", i, ":", rmse_test, "\n")
}

# Calculate average metrics across folds
mean_correlation_bayesb <- mean(sapply(cv_results_bayesb, function(res) res$correlation))
mean_r2_bayesb <- mean(sapply(cv_results_bayesb, function(res) res$r2))
mean_rmse_bayesb <- mean(sapply(cv_results_bayesb, function(res) res$rmse))

# Calculate global test set R²
global_ss_total <- sum((observed_all - mean(observed_all))^2)
global_ss_residual <- sum((observed_all - predicted_all)^2)
global_r2 <- 1 - (global_ss_residual / global_ss_total)

cat("\nMean Correlation Across 7 Folds (BayesB):", mean_correlation_bayesb, "\n")
cat("Mean R² Across 7 Folds (BayesB):", mean_r2_bayesb, "\n")
cat("Mean RMSE Across 7 Folds (BayesB):", mean_rmse_bayesb, "\n")
cat("Global R² Across All Test Sets (BayesB):", global_r2, "\n")

# Optional: Save results to file
write.csv(data.frame(
  Fold = seq_along(cv_results_bayesb),
  Correlation = sapply(cv_results_bayesb, function(res) res$correlation),
  R2 = sapply(cv_results_bayesb, function(res) res$r2),
  RMSE = sapply(cv_results_bayesb, function(res) res$rmse)
), "bayesb_cv_results.csv", row.names = FALSE)




# Load necessary libraries
library(glmnet)  # For Lasso
library(caret)   # For cross-validation

# Load genotype and phenotype data
# X <- as.matrix(read.csv("genotype_matrix.csv"))  # SNP matrix
y <- as.numeric(read.table("averaged_SW.txt", sep = '\t', header = T)$Average_SW)  # Phenotype vector

# Check for missing data and impute if necessary
X[is.na(X)] <- 0  # Replace missing SNPs with 0
y[is.na(y)] <- mean(y, na.rm = TRUE)  # Impute missing phenotypes with mean

# 7-Fold Cross-Validation Setup
set.seed(123)  # For reproducibility
folds <- createFolds(y, k = 7)

# Initialize list to store CV results and aggregated predictions
cv_results_lasso <- list()
predicted_all <- numeric(length(y))  # Store all test set predictions
observed_all <- numeric(length(y))  # Store all test set observations

# RMSE Function
rmse <- function(observed, predicted) {
  sqrt(mean((observed - predicted)^2))
}

# Perform 7-fold CV
for (i in seq_along(folds)) {
  cat("\nProcessing Fold:", i, "\n")
  
  # Training and test indices
  test_indices <- folds[[i]]
  train_indices <- setdiff(1:length(y), test_indices)
  
  # Training and test data
  y_train <- y[train_indices]
  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  
  # Fit Lasso Regression (alpha = 1 for Lasso)
  lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 5)  # Internal CV for lambda
  
  # Predict on test set using the best lambda
  GEBVs_test <- predict(lasso_model, X_test, s = "lambda.min")
  
  # Store test set predictions and observations
  predicted_all[test_indices] <- GEBVs_test
  observed_all[test_indices] <- y[test_indices]
  
  # Calculate performance metrics
  cor_test <- cor(y[test_indices], GEBVs_test)
  ss_total <- sum((y[test_indices] - mean(y[test_indices]))^2)
  ss_residual <- sum((y[test_indices] - GEBVs_test)^2)
  r2_test <- 1 - (ss_residual / ss_total)
  rmse_test <- rmse(y[test_indices], GEBVs_test)
  
  # Store results
  cv_results_lasso[[i]] <- list(
    fold = i,
    predicted = GEBVs_test,
    observed = y[test_indices],
    correlation = cor_test,
    r2 = r2_test,
    rmse = rmse_test
  )
  
  cat("Correlation for Fold", i, ":", cor_test, "\n")
  cat("R² for Fold", i, ":", r2_test, "\n")
  cat("RMSE for Fold", i, ":", rmse_test, "\n")
}

# Calculate average metrics across folds
mean_correlation_lasso <- mean(sapply(cv_results_lasso, function(res) res$correlation))
mean_r2_lasso <- mean(sapply(cv_results_lasso, function(res) res$r2))
mean_rmse_lasso <- mean(sapply(cv_results_lasso, function(res) res$rmse))

# Calculate global test set R²
global_ss_total <- sum((observed_all - mean(observed_all))^2)
global_ss_residual <- sum((observed_all - predicted_all)^2)
global_r2 <- 1 - (global_ss_residual / global_ss_total)

cat("\nMean Correlation Across 7 Folds (Lasso):", mean_correlation_lasso, "\n")
cat("Mean R² Across 7 Folds (Lasso):", mean_r2_lasso, "\n")
cat("Mean RMSE Across 7 Folds (Lasso):", mean_rmse_lasso, "\n")
cat("Global R² Across All Test Sets (Lasso):", global_r2, "\n")

# Optional: Save results to file
write.csv(data.frame(
  Fold = seq_along(cv_results_lasso),
  Correlation = sapply(cv_results_lasso, function(res) res$correlation),
  R2 = sapply(cv_results_lasso, function(res) res$r2),
  RMSE = sapply(cv_results_lasso, function(res) res$rmse)
), "lasso_cv_results.csv", row.names = FALSE)



# Load necessary libraries
library(e1071)  # For SVM
library(caret)  # For cross-validation

# Load genotype and phenotype data
# X <- as.matrix(read.csv("genotype_matrix.csv"))  # SNP matrix
y <- as.numeric(read.table("averaged_SW.txt", sep = '\t', header = T)$Average_SW)  # Phenotype vector

# Check for missing data and impute if necessary
X[is.na(X)] <- 0  # Replace missing SNPs with 0
y[is.na(y)] <- mean(y, na.rm = TRUE)  # Impute missing phenotypes with mean

# 7-Fold Cross-Validation Setup
set.seed(123)  # For reproducibility
folds <- createFolds(y, k = 7)

# Initialize list to store CV results and aggregated predictions
cv_results_svm <- list()
predicted_all <- numeric(length(y))  # Store all test set predictions
observed_all <- numeric(length(y))  # Store all test set observations

# RMSE Function
rmse <- function(observed, predicted) {
  sqrt(mean((observed - predicted)^2))
}

# Perform 7-fold CV
for (i in seq_along(folds)) {
  cat("\nProcessing Fold:", i, "\n")
  
  # Training and test indices
  test_indices <- folds[[i]]
  train_indices <- setdiff(1:length(y), test_indices)
  
  # Training and test data
  y_train <- y[train_indices]
  X_train <- X[train_indices, ]
  X_test <- X[test_indices, ]
  
  # Fit SVM model (default radial kernel)
  svm_model <- svm(X_train, y_train, type = "eps-regression", kernel = "radial", cost = 10, epsilon = 0.1)
  
  # Predict on test set
  GEBVs_test <- predict(svm_model, X_test)
  
  # Store test set predictions and observations
  predicted_all[test_indices] <- GEBVs_test
  observed_all[test_indices] <- y[test_indices]
  
  # Calculate performance metrics
  cor_test <- cor(y[test_indices], GEBVs_test)
  ss_total <- sum((y[test_indices] - mean(y[test_indices]))^2)
  ss_residual <- sum((y[test_indices] - GEBVs_test)^2)
  r2_test <- 1 - (ss_residual / ss_total)
  rmse_test <- rmse(y[test_indices], GEBVs_test)
  
  # Store results
  cv_results_svm[[i]] <- list(
    fold = i,
    predicted = GEBVs_test,
    observed = y[test_indices],
    correlation = cor_test,
    r2 = r2_test,
    rmse = rmse_test
  )
  
  cat("Correlation for Fold", i, ":", cor_test, "\n")
  cat("R² for Fold", i, ":", r2_test, "\n")
  cat("RMSE for Fold", i, ":", rmse_test, "\n")
}

# Calculate average metrics across folds
mean_correlation_svm <- mean(sapply(cv_results_svm, function(res) res$correlation))
mean_r2_svm <- mean(sapply(cv_results_svm, function(res) res$r2))
mean_rmse_svm <- mean(sapply(cv_results_svm, function(res) res$rmse))

# Calculate global test set R²
global_ss_total <- sum((observed_all - mean(observed_all))^2)
global_ss_residual <- sum((observed_all - predicted_all)^2)
global_r2 <- 1 - (global_ss_residual / global_ss_total)

cat("\nMean Correlation Across 7 Folds (SVM):", mean_correlation_svm, "\n")
cat("Mean R² Across 7 Folds (SVM):", mean_r2_svm, "\n")
cat("Mean RMSE Across 7 Folds (SVM):", mean_rmse_svm, "\n")
cat("Global R² Across All Test Sets (SVM):", global_r2, "\n")

# Optional: Save results to file
write.csv(data.frame(
  Fold = seq_along(cv_results_svm),
  Correlation = sapply(cv_results_svm, function(res) res$correlation),
  R2 = sapply(cv_results_svm, function(res) res$r2),
  RMSE = sapply(cv_results_svm, function(res) res$rmse)
), "svm_cv_results.csv", row.names = FALSE)






