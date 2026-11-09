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