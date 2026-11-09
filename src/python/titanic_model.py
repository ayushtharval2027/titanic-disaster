"""
Titanic survival prediction with logistic regression.

Loads the Kaggle Titanic data, cleans it, trains a logistic regression
model, and prints every step to the terminal.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Show all columns when printing DataFrames in the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
# This script lives in src/python/, and the data lives in src/data/.
# Locally:          src/python/titanic_model.py -> src/data/
# Inside Docker:    /app/titanic_model.py       -> /data/  (mounted with -v)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Step 1: Load the training data
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Load the training data")
print("=" * 70)

if not TRAIN_PATH.exists():
    sys.exit(
        f"ERROR: Could not find {TRAIN_PATH}\n"
        "Download train.csv and test.csv from Kaggle and place them in src/data/ "
        "(see README for instructions)."
    )

train = pd.read_csv(TRAIN_PATH)
print(f"Loaded {TRAIN_PATH}")
print(f"Shape: {train.shape[0]} rows x {train.shape[1]} columns\n")
print("First 5 rows:")
print(train.head())