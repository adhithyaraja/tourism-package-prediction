"""
=============================================================================
 STEP 2 : DATA PREPARATION  (tourism_project/model_building/prep.py)
=============================================================================
Purpose
-------
Second job of the pipeline. It reads the registered CSV straight from the
repository's data folder, cleans it, and produces a reproducible train/test
split that the training job consumes as a GitHub Actions *artifact*.

What it does
------------
  1. Load tourism_project/data/tourism.csv.
  2. Drop columns that carry no predictive signal:
       - 'Unnamed: 0' : leftover pandas index from the original export.
       - 'CustomerID' : a unique identifier; keeping it would let a tree model
                        memorise individual customers instead of learning
                        behaviour (a classic source of leakage/overfitting).
  3. Fix known data-quality issues (the 'Fe Male' typo in Gender).
  4. Drop exact duplicate rows.
  5. Stratified 80/20 train/test split so that the ~19% purchase rate is
     preserved in both halves.
  6. Write Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to the repository root
     so the workflow can upload them as an artifact for the next job.

Note on missing values
----------------------
Missing values are NOT imputed here. Imputation is fitted *inside* the
scikit-learn pipeline in train.py, so the imputation statistics are learned
from the training fold only. Doing it here would leak test information into
training.
=============================================================================
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Path resolution (same convention as data_register.py).
# Outputs go to the repository root because the workflow's upload-artifact step
# and the matching download-artifact step both operate from that directory.
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

TARGET_COLUMN = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    print("=" * 78)
    print("DATA PREPARATION")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Load the dataset from the repository data folder.
    # -------------------------------------------------------------------------
    print(f"\n[1/6] Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"      Loaded shape: {df.shape}")

    # -------------------------------------------------------------------------
    # 2. Drop the identifier / index columns.
    #    errors='ignore' keeps the script working even if a column is absent.
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Dropping non-predictive columns: {DROP_COLUMNS}")
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    print(f"      Shape after drop: {df.shape}")

    # -------------------------------------------------------------------------
    # 3. Clean known category typos.
    #    'Fe Male' is a data-entry variant of 'Female'. Left uncleaned, the
    #    one-hot encoder would create a third, meaningless gender column.
    # -------------------------------------------------------------------------
    print("\n[3/6] Cleaning categorical values")
    if "Gender" in df.columns:
        before = sorted(df["Gender"].dropna().unique().tolist())
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
        after = sorted(df["Gender"].dropna().unique().tolist())
        print(f"      Gender levels before: {before}")
        print(f"      Gender levels after : {after}")

    # -------------------------------------------------------------------------
    # 4. Remove exact duplicate records.
    # -------------------------------------------------------------------------
    dupes = int(df.duplicated().sum())
    print(f"\n[4/6] Removing duplicate rows (found {dupes})")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"      Shape after de-duplication: {df.shape}")

    # -------------------------------------------------------------------------
    # 5. Separate predictors (X) from the target (y) and split.
    #    stratify=y keeps the minority class proportion identical in both sets,
    #    which matters a lot for an imbalanced target like this one.
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Splitting into train/test (test_size={TEST_SIZE}, stratified)")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"      Xtrain: {Xtrain.shape}   ytrain positive rate: {ytrain.mean():.4f}")
    print(f"      Xtest : {Xtest.shape}   ytest  positive rate: {ytest.mean():.4f}")
    print(f"      Predictors used ({X.shape[1]}): {list(X.columns)}")

    # -------------------------------------------------------------------------
    # 6. Persist the four split files at the repository root.
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Saving split files to {REPO_ROOT}")
    outputs = {
        "Xtrain.csv": Xtrain,
        "Xtest.csv": Xtest,
        "ytrain.csv": ytrain,
        "ytest.csv": ytest,
    }
    for name, frame in outputs.items():
        path = REPO_ROOT / name
        frame.to_csv(path, index=False)
        print(f"      Saved {name:<11} -> {path}")

    print("\n" + "=" * 78)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
