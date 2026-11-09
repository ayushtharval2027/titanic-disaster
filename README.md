# Titanic Survival Prediction: Python and R in Docker

Predicts which passengers survived the Titanic disaster using logistic regression. The same pipeline is implemented twice, once in Python and once in R, and each runs in its own Docker container. Every step (loading, exploring, cleaning, training, evaluating, and predicting) prints its results to the terminal.

## Contents

- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Step 1: Clone the repository](#step-1-clone-the-repository)
- [Step 2: Download the data](#step-2-download-the-data)
- [Step 3: Run the Python container](#step-3-run-the-python-container)
- [Step 4: Run the R container](#step-4-run-the-r-container)
- [What the commands do](#what-the-commands-do)
- [What the scripts do](#what-the-scripts-do)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## Repository structure

```
titanic-disaster/
├── README.md
├── .gitignore                 # keeps the dataset and local files out of the repo
├── CODEOWNERS
└── src/
    ├── data/                  # NOT in the repo: you create it and add the CSVs (Step 2)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt   # pandas 3.1.1, scikit-learn 1.9.1
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R # dplyr 1.2.1
        └── titanic_model.R
```

The dataset is not included in this repository. Both containers read it from `src/data/`, which is connected to the container when it runs.

## Prerequisites

- **Git** to clone the repository ([download](https://git-scm.com/downloads)), or GitHub Desktop
- **Docker Desktop** ([download](https://docs.docker.com/get-started/get-docker/)), **open and running**
- A free **Kaggle account** to download the data

You do not need Python or R installed. Both run inside the containers.

Before starting, confirm Docker is running:

```bash
docker version
```

The output must include a **Server** section. If you see only a Client section followed by an error such as `failed to connect to the docker API`, open Docker Desktop and wait until it shows **Engine running**.

## Step 1: Clone the repository

```bash
git clone https://github.com/ayushtharval2027/titanic-disaster.git
cd titanic-disaster
```

Using GitHub Desktop instead: **File → Clone repository → URL**, paste `https://github.com/ayushtharval2027/titanic-disaster`, then open a terminal in the cloned folder.

**Run every command in the rest of this README from the `titanic-disaster` folder (the repository root).**

## Step 2: Download the data

1. Go to the [Titanic competition data page](https://www.kaggle.com/competitions/titanic/data) and sign in to Kaggle.
2. If prompted, join the competition and accept its rules. Kaggle requires this before allowing the download.
3. Click **Download All** to download `titanic.zip`.
4. Create the data folder and unzip the files into it.

**macOS / Linux (Terminal):**

```bash
mkdir -p src/data
unzip ~/Downloads/titanic.zip -d src/data
```

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force src/data
Expand-Archive "$HOME\Downloads\titanic.zip" -DestinationPath src\data
```

If your browser unzipped the download automatically, or it was saved somewhere other than Downloads, copy `train.csv` and `test.csv` into `src/data/` manually instead.

5. Confirm the files are in place:

```bash
ls src/data
```

You should see `train.csv` and `test.csv`. (`gender_submission.csv` may also be there; it is not used.)

## Step 3: Run the Python container

**Build the image** (the first build takes about 1–2 minutes):

```bash
docker build -t titanic-python src/python
```

**Run the container:**

macOS / Linux:

```bash
docker run --rm -v "$(pwd)/src/data:/data" titanic-python
```

Windows PowerShell:

```powershell
docker run --rm -v "${PWD}/src/data:/data" titanic-python
```

The only difference is how the current folder is written: `$(pwd)` on macOS/Linux, `${PWD}` in PowerShell.

**Expected output:** seven numbered steps, starting with:

```
======================================================================
STEP 1: Load the training data
======================================================================
Loaded /data/train.csv
Shape: 891 rows x 12 columns
```

and ending with the 418 test predictions followed by `Done.`

## Step 4: Run the R container

**Build the image** (the first build takes several minutes because it downloads R and installs dplyr):

```bash
docker build -t titanic-r src/r
```

The build log should include `Installed dplyr 1.2.1`.

**Run the container:**

macOS / Linux:

```bash
docker run --rm -v "$(pwd)/src/data:/data" titanic-r
```

Windows PowerShell:

```powershell
docker run --rm -v "${PWD}/src/data:/data" titanic-r
```

**Expected output:** the same seven steps as the Python container, starting with `Loaded /data/train.csv` and ending with `Done.`

## What the commands do

| Command or flag | Meaning |
|---|---|
| `docker build -t titanic-python src/python` | Builds an image named `titanic-python` using the `Dockerfile` in `src/python` |
| `docker run ... titanic-python` | Starts a container from that image; the script runs automatically |
| `--rm` | Deletes the container after the script finishes, so finished containers don't pile up |
| `-v "<current folder>/src/data:/data"` | Connects your local `src/data` folder to `/data` inside the container, so the script can read the CSVs without them being built into the image |

## What the scripts do

Both scripts follow the same seven steps and print the result of each one:

1. **Load** `train.csv` (891 passengers).
2. **Explore** data types, missing values, and survival rates by sex, class, and port.
3. **Clean** the data:
   - Fill missing `Age` and `Fare` with the training median, and missing `Embarked` with the most common port (S). Fill values are learned from the training data only and reused for the test data.
   - Add `FamilySize` = `SibSp` + `Parch` + 1.
   - Encode `Sex` as 0/1 and one-hot encode `Embarked`, with S as the baseline.
   - Drop `PassengerId`, `Name`, `Ticket`, `Cabin` (77% missing), `SibSp`, and `Parch`.
4. **Train** a logistic regression model on 7 features: `Pclass`, `Sex`, `Age`, `Fare`, `FamilySize`, `Embarked_C`, `Embarked_Q`.
5. **Measure training accuracy**, compared with a baseline that predicts everyone died.
6. **Load and clean** `test.csv` with the same steps and fill values.
7. **Print predictions** for all 418 test passengers.

`test.csv` has no `Survived` column, so accuracy on the test set cannot be measured. As instructed for this assignment, the predictions are printed instead.

## Results

| | Python | R |
|---|---|---|
| Base image | `python:3.14-slim` | `rocker/r-ver:4.6.1` |
| Model | scikit-learn `LogisticRegression` | `glm(family = binomial)` |
| Training accuracy | 80.13% | 79.91% |
| Baseline (predict everyone died) | 61.62% | 61.62% |
| Test predictions | 156 survived, 262 died | 156 survived, 262 died |

The coefficients differ slightly because scikit-learn applies L2 regularization by default and `glm` does not. This changes two borderline training predictions, but the 418 test predictions are identical in both containers. In both models, sex is by far the strongest predictor (being female multiplies the odds of survival by about 14–16), followed by passenger class.

## Troubleshooting

| Problem | Fix |
|---|---|
| `failed to connect to the docker API` or `Cannot connect to the Docker daemon` | Docker Desktop is not running. Open it and wait for **Engine running**. |
| `ERROR: Could not find /data/train.csv` | The data is not connected to the container. Check that `src/data/train.csv` exists (Step 2), that you are in the repository root, and that the command includes the `-v` part. |
| `docker build` says the path is not found | Run the command from the repository root, so `src/python` and `src/r` exist relative to your current folder. |
| On Windows in Git Bash, the data is not found even though the files exist | Git Bash rewrites paths in the `-v` flag. Use PowerShell instead, or put `MSYS_NO_PATHCONV=1` before the `docker run` command. |
| The R build takes several minutes | Normal for the first build. Later builds reuse cached layers and finish in seconds. |