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