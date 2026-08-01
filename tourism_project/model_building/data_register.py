"""
=============================================================================
 STEP 1 : DATA REGISTRATION  (tourism_project/model_building/data_register.py)
=============================================================================
Purpose
-------
This is the first job of the MLOps pipeline. It "registers" the raw dataset
that lives inside the GitHub repository (tourism_project/data/tourism.csv).

Registration here means *validating and documenting* the data contract before
any downstream job is allowed to run:

  1. Confirm the CSV physically exists in the repo.
  2. Confirm every column named in the data dictionary is present
     (schema validation -> fails the pipeline early if the contract breaks).
  3. Print a human-readable summary (shape, dtypes, nulls, target balance)
     so the GitHub Actions log itself becomes the dataset audit record.

Why this matters in MLOps
-------------------------
If an upstream team silently renames or drops a column, we want the pipeline
to fail *here* with a clear message, not three jobs later inside a confusing
scikit-learn traceback. A non-zero exit code stops the whole workflow.
=============================================================================
"""

import sys
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Resolve paths relative to THIS file, never relative to the current working
# directory. This makes the script behave identically when it is run from
# Google Colab, from a laptop, or from the GitHub Actions runner.
#   parents[0] -> model_building/
#   parents[1] -> tourism_project/
#   parents[2] -> repository root
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tourism.csv"

# -----------------------------------------------------------------------------
# The data contract: every column listed in the project's data dictionary.
# 'Unnamed: 0' is deliberately NOT listed - it is a stray pandas index column
# that gets dropped during preparation, so it is optional, not required.
# -----------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

TARGET_COLUMN = "ProdTaken"

# Listed explicitly rather than sniffed with select_dtypes, so the report is
# identical across pandas versions.
CATEGORICAL_COLUMNS = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]


def main() -> None:
    print("=" * 78)
    print("DATA REGISTRATION - Visit with Us : Wellness Tourism Package")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Existence check - the dataset must be committed inside the repository.
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Looking for the dataset at: {DATA_PATH}")
    if not DATA_PATH.exists():
        print("      FAILED: tourism.csv was not found in tourism_project/data/.")
        print("      Commit the CSV to that folder and re-run the pipeline.")
        sys.exit(1)
    print(f"      OK - file found ({DATA_PATH.stat().st_size / 1024:.1f} KB).")

    # -------------------------------------------------------------------------
    # 2. Load the dataset.
    # -------------------------------------------------------------------------
    print("\n[2/4] Loading the dataset into a pandas DataFrame ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      OK - loaded {df.shape[0]} rows x {df.shape[1]} columns.")

    # -------------------------------------------------------------------------
    # 3. Schema validation - the core of "registering" the dataset.
    #    Missing expected columns is fatal. Extra columns are only a warning,
    #    because the preparation step is allowed to drop them.
    # -------------------------------------------------------------------------
    print("\n[3/4] Validating the schema against the data dictionary ...")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing:
        print(f"      FAILED: the following expected columns are missing -> {missing}")
        sys.exit(1)
    print(f"      OK - all {len(EXPECTED_COLUMNS)} expected columns are present.")

    if extra:
        print(f"      NOTE: extra column(s) found and will be dropped later -> {extra}")

    if TARGET_COLUMN not in df.columns:
        print(f"      FAILED: target column '{TARGET_COLUMN}' is missing.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Summary report - printed into the Actions log as the audit record.
    # -------------------------------------------------------------------------
    print("\n[4/4] Dataset summary")
    print("-" * 78)
    print(f"Rows                 : {df.shape[0]}")
    print(f"Columns              : {df.shape[1]}")
    print(f"Duplicate rows       : {int(df.duplicated().sum())}")
    print(f"Duplicate CustomerID : {int(df['CustomerID'].duplicated().sum())}")
    print(f"Total missing values : {int(df.isnull().sum().sum())}")

    print("\nColumn data types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string() if nulls.sum() else "  None - the dataset is complete.")

    print(f"\nTarget distribution ('{TARGET_COLUMN}'):")
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    shares = df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    for label in counts.index:
        meaning = "did NOT purchase" if label == 0 else "PURCHASED the package"
        print(f"  {label} ({meaning:<22}) : {counts[label]:>5}  ({shares[label]:.2%})")
    print("  -> The classes are imbalanced; the training step compensates for this.")

    print("\nNumeric summary:")
    print(df.describe().T.to_string())

    print("\nCategorical levels:")
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            print(f"  {col:<16}: {sorted(df[col].dropna().unique().tolist())}")

    print("\n" + "=" * 78)
    print("DATA REGISTRATION COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
