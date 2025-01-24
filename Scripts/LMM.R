# Load necessary libraries
library(lme4)    # For linear mixed models
library(dplyr)   # For data manipulation
library(ggplot2) # For visualization

# Set working directory (update path accordingly)
setwd("/Users/thulanihewavithana/Documents/PhD/Courses/CMPT 898/CMPT-PLSC_819_Project/GxE/SW/")

# Load datasets
phenotypic_data <- read.csv("Phenotype_SW.csv", header = TRUE)
genotypic_data <- read.csv("genotype_admixture.csv", header = TRUE)
environmental_data <- read.csv("Seeding_Date_Environmental_Averages.csv", header = TRUE)
train_data_merged <- read.csv("train_data_with_feature_map.csv", header = TRUE)
test_data_merged <- read.csv("test_data_with_feature_map.csv", header = TRUE)

# Merge training data
merged_data_train <- phenotypic_data %>%
  inner_join(train_data_merged, by = c("Sample_ID", "Location", "Year", "Reps", "SD")) %>%
  inner_join(environmental_data, by = c("Location", "Year", "SD")) %>%
  inner_join(genotypic_data, by = "Sample_ID")

# Merge test data
merged_data_test <- phenotypic_data %>%
  inner_join(test_data_merged, by = c("Sample_ID", "Location", "Year", "Reps", "SD")) %>%
  inner_join(environmental_data, by = c("Location", "Year", "SD")) %>%
  inner_join(genotypic_data, by = "Sample_ID")

# Scale continuous predictors
merged_data_train <- merged_data_train %>%
  mutate(across(c(SW.y, Avg.Max.Temp, Avg.Precipitation..Literature), scale))

merged_data_test <- merged_data_test %>%
  mutate(across(c(SW.y, Avg.Max.Temp, Avg.Precipitation..Literature), scale))

# Define the LMM formula with interaction terms
formula <- True_Label ~ SW.y * Avg.Max.Temp + SW.y * Avg.Precipitation..Literature +
  (1 | Location) + (1 | Year) + (1 | Sample_ID:Avg.Max.Temp) + (1 | Sample_ID:Avg.Precipitation..Literature)

# Custom function for LMM Cross-Validation
lmm_cross_validation <- function(data, formula, folds = 7) {
  set.seed(42) # For reproducibility
  data <- data %>% mutate(Fold = sample(rep(1:folds, length.out = n())))
  
  mse_values <- c()
  r_squared_values <- c()
  
  for (fold in 1:folds) {
    train_data <- data %>% filter(Fold != fold)
    val_data <- data %>% filter(Fold == fold)
    
    # Fit the model on the training data
    model <- lmer(formula, data = train_data, control = lmerControl(optimizer = "bobyqa"))
    
    # Predict on the validation data
    val_data$predictions <- predict(model, newdata = val_data, allow.new.levels = TRUE)
    
    # Compute metrics
    mse <- mean((val_data$True_Label - val_data$predictions)^2, na.rm = TRUE)
    correlation <- cor(val_data$True_Label, val_data$predictions, use = "complete.obs")
    r_squared <- correlation^2
    
    mse_values <- c(mse_values, mse)
    r_squared_values <- c(r_squared_values, r_squared)
  }
  
  # Return average metrics across folds
  list(
    Mean_MSE = mean(mse_values),
    Mean_R2 = mean(r_squared_values),
    Fold_MSEs = mse_values,
    Fold_R2s = r_squared_values
  )
}

# Perform 7-fold Cross-Validation on training data
cv_results <- lmm_cross_validation(merged_data_train, formula, folds = 7)

# Print Cross-Validation Results
cat("Cross-Validated Mean MSE:", cv_results$Mean_MSE, "\n")
cat("Cross-Validated Mean R²:", cv_results$Mean_R2, "\n")

# Plot Cross-Validation Results
folds <- seq_along(cv_results$Fold_MSEs)

mse_df <- data.frame(Fold = folds, MSE = cv_results$Fold_MSEs)
r2_df <- data.frame(Fold = folds, R2 = cv_results$Fold_R2s)

ggplot(mse_df, aes(x = Fold, y = MSE)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Cross-Validation MSE for Each Fold", x = "Fold", y = "MSE")

ggplot(r2_df, aes(x = Fold, y = R2)) +
  geom_bar(stat = "identity", fill = "darkorange") +
  labs(title = "Cross-Validation R² for Each Fold", x = "Fold", y = "R²")

# Train Final Model on Entire Training Data
final_model <- lmer(formula, data = merged_data_train, control = lmerControl(optimizer = "bobyqa"))

# Predict on Test Data
merged_data_test$predictions <- predict(final_model, newdata = merged_data_test, allow.new.levels = TRUE)

# Compute Test Metrics
mse_test <- mean((merged_data_test$True_Label - merged_data_test$predictions)^2, na.rm = TRUE)
correlation_test <- cor(merged_data_test$True_Label, merged_data_test$predictions, use = "complete.obs")
r_squared_test <- correlation_test^2

cat("Test Data Mean Squared Error (MSE):", mse_test, "\n")
cat("Test Data R²:", r_squared_test, "\n")

mean_actual_test <- mean(merged_data_test$True_Label, na.rm = TRUE)
mae_test <- mean(abs(merged_data_test$True_Label - merged_data_test$predictions), na.rm = TRUE)
accuracy_test <- 100 * (1 - (mae_test / mean_actual_test))
correlation_test <- cor(merged_data_test$True_Label, merged_data_test$predictions, use = "complete.obs")
r_squared_test <- correlation_test^2

# Print Test Metrics
cat("Test Data Mean Squared Error (MSE):", mse_test, "\n")
cat("Test Data Mean Absolute Error (MAE):", mae_test, "\n")
cat("Test Data Accuracy (%):", accuracy_test, "\n")
cat("Test Data R²:", r_squared_test, "\n")

# Plot Actual vs Predicted for Test Data
ggplot(merged_data_test, aes(x = True_Label, y = predictions)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "blue") +
  labs(title = "Test Data: Actual vs Predicted", x = "True Label", y = "Predicted Label")

