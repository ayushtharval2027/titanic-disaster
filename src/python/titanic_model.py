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

# ---------------------------------------------------------------------------
# Step 2: Explore the training data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Explore the training data")
print("=" * 70)

print("\nColumn data types:")
print(train.dtypes)

print("\nMissing values per column:")
missing = train.isna().sum()
missing_pct = (missing / len(train) * 100).round(1)
print(pd.DataFrame({"missing": missing, "percent": missing_pct}))

print(f"\nOverall survival rate: {train['Survived'].mean():.1%}")

print("\nSurvival rate by sex:")
print(train.groupby("Sex")["Survived"].mean().round(3))

print("\nSurvival rate by passenger class:")
print(train.groupby("Pclass")["Survived"].mean().round(3))

print("\nSurvival rate by port of embarkation:")
print(train.groupby("Embarked")["Survived"].mean().round(3))

# ---------------------------------------------------------------------------
# Step 3: Clean and adjust the data
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Clean and adjust the data")
print("=" * 70)

# Fill values are learned from the TRAINING data only, then reused on the
# test data in Step 6, so both datasets are cleaned identically.
fill_values = {
    "Age": train["Age"].median(),
    "Fare": train["Fare"].median(),
    "Embarked": train["Embarked"].mode()[0],
}
print("\nFill values learned from the training data:")
for column, value in fill_values.items():
    print(f"  {column}: {value}")

# Final list of columns the model will use
FEATURES = ["Pclass", "Sex", "Age", "Fare", "FamilySize", "Embarked_C", "Embarked_Q"]


def prepare_data(df, fill_values):
    """Clean a raw Titanic DataFrame and return the model-ready features."""
    df = df.copy()

    # 1. Fill missing values
    for column, value in fill_values.items():
        n_missing = df[column].isna().sum()
        df[column] = df[column].fillna(value)
        print(f"  Filled {n_missing} missing '{column}' values with {value}")

    # 2. Add a new feature: family size = siblings/spouses + parents/children + self
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print("  Added 'FamilySize' = SibSp + Parch + 1")

    # 3. Encode Sex as a number: male = 0, female = 1
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("  Encoded 'Sex' as male = 0, female = 1")

    # 4. One-hot encode Embarked into 0/1 columns (S is the baseline)
    for port in ["C", "Q"]:
        df[f"Embarked_{port}"] = (df["Embarked"] == port).astype(int)
    print("  One-hot encoded 'Embarked' into Embarked_C, Embarked_Q (S = baseline)")

    # 5. Drop columns that are not used as features
    dropped = [c for c in df.columns if c not in FEATURES + ["Survived"]]
    df = df.drop(columns=dropped)
    print(f"  Dropped columns: {dropped}")

    return df


print("\nCleaning the training data:")
train_clean = prepare_data(train, fill_values)

print(f"\nCleaned training data: {train_clean.shape[0]} rows x {train_clean.shape[1]} columns")
print(f"Missing values remaining: {train_clean.isna().sum().sum()}")
print("\nFirst 5 rows after cleaning:")
print(train_clean.head())

# ---------------------------------------------------------------------------
# Step 4: Train a logistic regression model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Train a logistic regression model")
print("=" * 70)

X_train = train_clean[FEATURES]
y_train = train_clean["Survived"]
print(f"\nFeatures (X): {FEATURES}")
print(f"Target (y): Survived  ->  {X_train.shape[0]} passengers")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(f"\nModel trained (converged in {model.n_iter_[0]} iterations)")

coefficients = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_[0].round(4),
}).sort_values("coefficient", ascending=False)
print(f"\nIntercept: {model.intercept_[0]:.4f}")
print("Coefficients (positive = higher chance of survival):")
print(coefficients.to_string(index=False))

# ---------------------------------------------------------------------------
# Step 5: Measure accuracy on the training set
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Measure accuracy on the training set")
print("=" * 70)

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
baseline_accuracy = 1 - y_train.mean()

print(f"\nTraining accuracy: {train_accuracy:.2%}")
print(f"  ({(train_predictions == y_train).sum()} of {len(y_train)} passengers predicted correctly)")
print(f"Baseline accuracy (predict everyone died): {baseline_accuracy:.2%}")
print(f"Improvement over baseline: {(train_accuracy - baseline_accuracy) * 100:+.2f} percentage points")

labels = {0: "Died", 1: "Survived"}
results = pd.DataFrame({
    "Actual": y_train.map(labels),
    "Predicted": pd.Series(train_predictions, index=y_train.index).map(labels),
})
print("\nPredicted vs. actual outcomes on the training set:")
print(pd.crosstab(results["Actual"], results["Predicted"]))