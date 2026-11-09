# titanic_model.R
# Titanic survival prediction with logistic regression (R version).
#
# Loads the Kaggle Titanic data, cleans it, trains a logistic regression
# model with glm(), and prints every step to the terminal.

suppressPackageStartupMessages(library(dplyr))

# Allow wider output so data frames wrap less in the terminal
options(width = 120)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
# Inside Docker, the Dockerfile sets DATA_DIR=/data (mounted with -v).
# Run locally from the repo root, DATA_DIR is not set, so it falls back to src/data.
data_dir <- Sys.getenv("DATA_DIR", unset = file.path("src", "data"))
train_path <- file.path(data_dir, "train.csv")
test_path <- file.path(data_dir, "test.csv")

# Print a section header, matching the Python script's output style
print_header <- function(title) {
  cat("\n", strrep("=", 70), "\n", title, "\n", strrep("=", 70), "\n", sep = "")
}

# Stop with a clear message if a data file is missing
check_file <- function(path) {
  if (!file.exists(path)) {
    cat("ERROR: Could not find ", path, "\n",
        "Download train.csv and test.csv from Kaggle and place them in src/data/ ",
        "(see README for instructions).\n", sep = "", file = stderr())
    quit(status = 1)
  }
}

# ---------------------------------------------------------------------------
# Step 1: Load the training data
# ---------------------------------------------------------------------------
print_header("STEP 1: Load the training data")

check_file(train_path)

# na.strings: treat empty cells as missing (NA), not as empty text ""
train <- read.csv(train_path, na.strings = c("", "NA"))
cat("Loaded", train_path, "\n")
cat("Shape:", nrow(train), "rows x", ncol(train), "columns\n\n")
cat("First 5 rows:\n")
print(head(train, 5))

# ---------------------------------------------------------------------------
# Step 2: Explore the training data
# ---------------------------------------------------------------------------
print_header("STEP 2: Explore the training data")

cat("\nColumn data types:\n")
print(data.frame(type = sapply(train, class)))

cat("\nMissing values per column:\n")
missing <- colSums(is.na(train))
print(data.frame(missing = missing, percent = round(missing / nrow(train) * 100, 1)))

cat(sprintf("\nOverall survival rate: %.1f%%\n", mean(train$Survived) * 100))

cat("\nSurvival rate by sex:\n")
train %>%
  group_by(Sex) %>%
  summarise(survival_rate = round(mean(Survived), 3)) %>%
  as.data.frame() %>%
  print()

cat("\nSurvival rate by passenger class:\n")
train %>%
  group_by(Pclass) %>%
  summarise(survival_rate = round(mean(Survived), 3)) %>%
  as.data.frame() %>%
  print()

cat("\nSurvival rate by port of embarkation:\n")
train %>%
  filter(!is.na(Embarked)) %>%
  group_by(Embarked) %>%
  summarise(survival_rate = round(mean(Survived), 3)) %>%
  as.data.frame() %>%
  print()
  
# ---------------------------------------------------------------------------
# Step 3: Clean and adjust the data
# ---------------------------------------------------------------------------
print_header("STEP 3: Clean and adjust the data")

# Fill values are learned from the TRAINING data only, then reused on the
# test data in Step 6, so both datasets are cleaned identically.
fill_values <- list(
  Age = median(train$Age, na.rm = TRUE),
  Fare = median(train$Fare, na.rm = TRUE),
  Embarked = names(which.max(table(train$Embarked)))
)
cat("\nFill values learned from the training data:\n")
for (column in names(fill_values)) {
  cat("  ", column, ": ", fill_values[[column]], "\n", sep = "")
}

# Final list of columns the model will use
FEATURES <- c("Pclass", "Sex", "Age", "Fare", "FamilySize", "Embarked_C", "Embarked_Q")

prepare_data <- function(df, fill_values) {
  # 1. Fill missing values
  for (column in names(fill_values)) {
    n_missing <- sum(is.na(df[[column]]))
    df[[column]][is.na(df[[column]])] <- fill_values[[column]]
    cat("  Filled ", n_missing, " missing '", column, "' values with ",
        fill_values[[column]], "\n", sep = "")
  }

  # 2. Add a new feature: family size = siblings/spouses + parents/children + self
  df <- df %>% mutate(FamilySize = SibSp + Parch + 1)
  cat("  Added 'FamilySize' = SibSp + Parch + 1\n")

  # 3. Encode Sex as a number: male = 0, female = 1
  df <- df %>% mutate(Sex = as.integer(Sex == "female"))
  cat("  Encoded 'Sex' as male = 0, female = 1\n")

  # 4. One-hot encode Embarked into 0/1 columns (S is the baseline)
  df <- df %>% mutate(
    Embarked_C = as.integer(Embarked == "C"),
    Embarked_Q = as.integer(Embarked == "Q")
  )
  cat("  One-hot encoded 'Embarked' into Embarked_C, Embarked_Q (S = baseline)\n")

  # 5. Drop columns that are not used as features
  dropped <- setdiff(names(df), c(FEATURES, "Survived"))
  df <- df %>% select(-all_of(dropped))
  cat("  Dropped columns:", paste(dropped, collapse = ", "), "\n")

  df
}

cat("\nCleaning the training data:\n")
train_clean <- prepare_data(train, fill_values)

cat("\nCleaned training data:", nrow(train_clean), "rows x", ncol(train_clean), "columns\n")
cat("Missing values remaining:", sum(is.na(train_clean)), "\n")
cat("\nFirst 5 rows after cleaning:\n")
print(head(train_clean, 5))

# ---------------------------------------------------------------------------
# Step 4: Train a logistic regression model
# ---------------------------------------------------------------------------
print_header("STEP 4: Train a logistic regression model")

# Build the model formula from the feature list: Survived ~ Pclass + Sex + ...
formula_text <- paste("Survived ~", paste(FEATURES, collapse = " + "))
cat("\nFeatures (X):", paste(FEATURES, collapse = ", "), "\n")
cat("Target (y): Survived  -> ", nrow(train_clean), "passengers\n")
cat("Model formula:", formula_text, "\n")

model <- glm(as.formula(formula_text), data = train_clean, family = binomial)
cat("\nModel trained (converged:", model$converged, "in", model$iter, "iterations)\n")

coefficients <- coef(model)
cat(sprintf("\nIntercept: %.4f\n", coefficients[["(Intercept)"]]))
coef_table <- data.frame(
  feature = FEATURES,
  coefficient = round(coefficients[FEATURES], 4)
)
coef_table <- coef_table[order(-coef_table$coefficient), ]
cat("Coefficients (positive = higher chance of survival):\n")
print(coef_table, row.names = FALSE)

# ---------------------------------------------------------------------------
# Step 5: Measure accuracy on the training set
# ---------------------------------------------------------------------------
print_header("STEP 5: Measure accuracy on the training set")

# type = "response" returns probabilities (0 to 1), not log-odds
train_probabilities <- predict(model, newdata = train_clean, type = "response")
train_predictions <- as.integer(train_probabilities >= 0.5)

n_correct <- sum(train_predictions == train_clean$Survived)
train_accuracy <- mean(train_predictions == train_clean$Survived)
baseline_accuracy <- 1 - mean(train_clean$Survived)

cat(sprintf("\nTraining accuracy: %.2f%%\n", train_accuracy * 100))
cat(sprintf("  (%d of %d passengers predicted correctly)\n", n_correct, nrow(train_clean)))
cat(sprintf("Baseline accuracy (predict everyone died): %.2f%%\n", baseline_accuracy * 100))
cat(sprintf("Improvement over baseline: %+.2f percentage points\n",
            (train_accuracy - baseline_accuracy) * 100))

labels <- c("0" = "Died", "1" = "Survived")
cat("\nPredicted vs. actual outcomes on the training set:\n")
print(table(
  Actual = labels[as.character(train_clean$Survived)],
  Predicted = labels[as.character(train_predictions)]
))

# ---------------------------------------------------------------------------
# Step 6: Load and clean the test data, then predict
# ---------------------------------------------------------------------------
print_header("STEP 6: Load the test data and make predictions")

check_file(test_path)

test <- read.csv(test_path, na.strings = c("", "NA"))
cat("Loaded", test_path, "\n")
cat("Shape:", nrow(test), "rows x", ncol(test), "columns\n")
cat("'Survived' column present:", "Survived" %in% names(test), "\n")

cat("\nMissing values in the test data:\n")
test_missing <- colSums(is.na(test))
print(test_missing[test_missing > 0])

cat("\nCleaning the test data with the SAME steps and training fill values:\n")
test_clean <- prepare_data(test, fill_values)

cat("\nCleaned test data:", nrow(test_clean), "rows x", ncol(test_clean), "columns\n")
cat("Missing values remaining:", sum(is.na(test_clean)), "\n")
cat("Columns match the training features:", identical(names(test_clean), FEATURES), "\n")

test_probabilities <- predict(model, newdata = test_clean, type = "response")
test_predictions <- as.integer(test_probabilities >= 0.5)
cat("\nPredicted survival for", length(test_predictions), "test passengers\n")

# ---------------------------------------------------------------------------
# Step 7: Print the test set predictions
# ---------------------------------------------------------------------------
# test.csv has no Survived column, so test accuracy cannot be measured.
# Per the course instructions, the predictions are printed instead.
print_header("STEP 7: Test set predictions")

predictions <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_predictions
)

n_survived <- sum(predictions$Survived)
n_total <- nrow(predictions)
cat("\nSummary:\n")
cat(sprintf("  Predicted survived: %d of %d (%.1f%%)\n",
            n_survived, n_total, n_survived / n_total * 100))
cat(sprintf("  Predicted died:     %d of %d (%.1f%%)\n",
            n_total - n_survived, n_total, (n_total - n_survived) / n_total * 100))

cat("\nPredicted survival rate by sex (test set):\n")
predictions %>%
  mutate(Sex = test$Sex) %>%
  group_by(Sex) %>%
  summarise(predicted_survival_rate = round(mean(Survived), 3)) %>%
  as.data.frame() %>%
  print()

cat("\nAll predictions (1 = survived, 0 = died):\n")
print(predictions, row.names = FALSE)

print_header("Done.")