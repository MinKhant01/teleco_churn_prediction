# ==============================================================================
# Script: 01_model_training.R
# Purpose: Train and evaluate Decision Tree and Random Forest models using tidymodels
# ==============================================================================

# 1. Load dependencies
library(tidymodels)   # Modern meta-package for ML modeling and preprocessing
library(rpart.plot)   # For visualizing the decision tree
library(ranger)       # Fast implementation of Random Forests

# 2. Load and clean data
telco <- read.csv("data/raw/Telco_customer_churn.csv")

# Remove ID, constants, and data leakage variables
telco <- telco |>
  dplyr::select(-CustomerID, -Count, -Country, -State, 
                -Churn.Value, -Churn.Score, -Churn.Reason)

# Handle missing Total Charges and recode redundant character strings
telco <- telco |>
  dplyr::mutate(
    Total.Charges = ifelse(is.na(Total.Charges), 0, Total.Charges),
    Online.Security = ifelse(Online.Security == "No internet service", "No", Online.Security),
    Online.Backup = ifelse(Online.Backup == "No internet service", "No", Online.Backup),
    Device.Protection = ifelse(Device.Protection == "No internet service", "No", Device.Protection),
    Tech.Support = ifelse(Tech.Support == "No internet service", "No", Tech.Support),
    Streaming.TV = ifelse(Streaming.TV == "No internet service", "No", Streaming.TV),
    Streaming.Movies = ifelse(Streaming.Movies == "No internet service", "No", Streaming.Movies),
    Multiple.Lines = ifelse(Multiple.Lines == "No phone service", "No", Multiple.Lines),
    # Ensure target variable is a factor
    Churn.Label = as.factor(Churn.Label)
  )

# Convert all remaining character columns to factors
telco <- telco |>
  dplyr::mutate_if(is.character, as.factor)

# ==============================================================================
# 3. Data Splitting and Preprocessing Recipe
# ==============================================================================
set.seed(1234)
# 80/20 Stratified split to maintain churn proportion
telco_split <- initial_split(telco, prop = 0.80, strata = Churn.Label)
train_data  <- training(telco_split)
test_data   <- testing(telco_split)

# Build the preprocessing recipe
telco_rec <- recipe(Churn.Label ~ ., data = train_data) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# ==============================================================================
# 4. Decision Tree Modeling (rpart engine)
# ==============================================================================
# Define the model specification
tree_spec <- decision_tree() |>
  set_engine("rpart") |>
  set_mode("classification")

# Create a workflow, add recipe and model, and fit to training data
tree_wf <- workflow() |>
  add_recipe(telco_rec) |>
  add_model(tree_spec)

tree_fit <- tree_wf |> fit(data = train_data)

# Extract and programmatically save the decision tree plot
png("outputs/figures/01_decision_tree.png", width = 1200, height = 800, res = 100)
rpart.plot(extract_fit_engine(tree_fit), 
           main = "Decision Tree for Telco Churn", 
           type = 4, extra = 104, fallen.leaves = TRUE)
dev.off()

# ==============================================================================
# 5. Random Forest Modeling (ranger engine)
# ==============================================================================
# Define the model specification with 100 trees and impurity-based importance
rf_spec <- rand_forest(trees = 100) |>
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification")

# Create workflow and fit to training data
rf_wf <- workflow() |>
  add_recipe(telco_rec) |>
  add_model(rf_spec)

set.seed(1234)
rf_fit <- rf_wf |> fit(data = train_data)

# Extract and programmatically save the Variable Importance plot
png("outputs/figures/02_rf_var_importance.png", width = 800, height = 600, res = 100)

# Extract raw importance scores from the ranger engine
rf_engine <- extract_fit_engine(rf_fit)
rf_importance <- sort(rf_engine$variable.importance, decreasing = FALSE)

# Plot the top 15 features 
par(mar = c(5, 12, 4, 2)) # Widen left margin to fit feature names
dotchart(tail(rf_importance, 15), 
         main = "Random Forest Variable Importance (Impurity)",
         pch = 16, col = "darkblue", lcolor = "gray90",
         xlab = "Mean Decrease Gini")

dev.off()

# ==============================================================================
# 6. Evaluation and Metrics
# ==============================================================================
# Generate predictions on the test set for both models
tree_preds <- augment(tree_fit, new_data = test_data)
rf_preds <- augment(rf_fit, new_data = test_data)

# Calculate and print metrics (Accuracy and ROC AUC)
cat("\n--- Decision Tree Performance ---\n")
print(tree_preds |> metrics(truth = Churn.Label, estimate = .pred_class))
print(tree_preds |> roc_auc(truth = Churn.Label, .pred_Yes))

cat("\n--- Random Forest Performance ---\n")
print(rf_preds |> metrics(truth = Churn.Label, estimate = .pred_class))
print(rf_preds |> roc_auc(truth = Churn.Label, .pred_Yes))