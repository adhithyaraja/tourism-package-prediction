"""
=============================================================================
 STEP 3 : MODEL BUILDING & EXPERIMENT TRACKING
          (tourism_project/model_building/train.py)
=============================================================================
Purpose
-------
Third job of the pipeline. It consumes the train/test artifact produced by
prep.py, tunes an XGBoost classifier, tracks every experiment in MLflow, and
saves the winning model so the workflow can commit it back into the repo -
which is exactly what the Streamlit app loads at serving time.

Pipeline design
---------------
Preprocessing and the estimator live inside ONE scikit-learn Pipeline object:

    ColumnTransformer
        numeric      -> SimpleImputer(median) -> StandardScaler
        categorical  -> SimpleImputer(most_frequent) -> OneHotEncoder
    XGBClassifier

Bundling them together means:
  * No leakage - imputation/scaling/encoding statistics are learned on the
    training fold only, inside every cross-validation split.
  * One artifact - the .joblib file contains preprocessing AND the model, so
    the Streamlit app can feed it a raw, human-entered DataFrame.

Class imbalance
---------------
Only ~19% of customers buy the package. `scale_pos_weight = neg/pos` tells
XGBoost to weight the minority class up, and the grid search optimises F1
rather than accuracy - a model that predicts "nobody buys" would score 81%
accuracy while being commercially useless.
=============================================================================
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

# -----------------------------------------------------------------------------
# Paths. The split CSVs sit at the repository root - that is where prep.py
# writes them locally and where download-artifact restores them in Actions.
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_DIR / "deployment"
MODEL_PATH = DEPLOY_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = DEPLOY_DIR / "model_metrics.json"

RANDOM_STATE = 42

# Feature groups. Binary flags (Passport/OwnCar) and ordinal scores (CityTier,
# PreferredPropertyStar, PitchSatisfactionScore) are treated as numeric because
# their ordering carries real meaning.
NUMERIC_FEATURES = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]


def configure_mlflow() -> str:
    """
    Point MLflow at the tracking server started by the workflow
    (`mlflow ui --port 5000`). If that server is unreachable - for example when
    somebody runs this script on a laptop - fall back to a local ./mlruns file
    store so training never fails just because tracking is unavailable.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("tourism-package-prediction")
        print(f"      MLflow tracking server: {uri}")
        return uri
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        fallback = (REPO_ROOT / "mlruns").as_uri()
        print(f"      Could not reach {uri} ({exc.__class__.__name__}).")
        print(f"      Falling back to local file store: {fallback}")
        mlflow.set_tracking_uri(fallback)
        mlflow.set_experiment("tourism-package-prediction")
        return fallback


def evaluate(model, X, y, split_name: str) -> dict:
    """Compute the standard classification metric set for one data split."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        f"{split_name}_accuracy": accuracy_score(y, preds),
        f"{split_name}_precision": precision_score(y, preds, zero_division=0),
        f"{split_name}_recall": recall_score(y, preds, zero_division=0),
        f"{split_name}_f1": f1_score(y, preds, zero_division=0),
        f"{split_name}_roc_auc": roc_auc_score(y, proba),
    }
    print(f"\n      --- {split_name.upper()} performance ---")
    for key, value in metrics.items():
        print(f"      {key:<22}: {value:.4f}")
    print(f"\n      Confusion matrix ({split_name}):\n{confusion_matrix(y, preds)}")
    print(f"\n      Classification report ({split_name}):\n{classification_report(y, preds, zero_division=0)}")
    return metrics


def main() -> None:
    print("=" * 78)
    print("MODEL BUILDING WITH EXPERIMENT TRACKING")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Load the train/test splits handed over by the previous job.
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading train/test splits from the workflow artifact")
    Xtrain = pd.read_csv(REPO_ROOT / "Xtrain.csv")
    Xtest = pd.read_csv(REPO_ROOT / "Xtest.csv")
    # .squeeze() turns the single-column frames back into Series.
    ytrain = pd.read_csv(REPO_ROOT / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(REPO_ROOT / "ytest.csv").squeeze("columns")
    print(f"      Xtrain {Xtrain.shape} | Xtest {Xtest.shape}")
    print(f"      Train positive rate {ytrain.mean():.4f} | Test positive rate {ytest.mean():.4f}")

    # -------------------------------------------------------------------------
    # 2. Configure experiment tracking.
    # -------------------------------------------------------------------------
    print("\n[2/7] Configuring MLflow")
    configure_mlflow()

    # -------------------------------------------------------------------------
    # 3. Build the preprocessing + model pipeline.
    # -------------------------------------------------------------------------
    print("\n[3/7] Building the preprocessing + model pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        # handle_unknown='ignore' keeps the app from crashing if
                        # a category it has never seen arrives at serving time.
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # Weight the minority class in proportion to how rare it is.
    neg, pos = int((ytrain == 0).sum()), int((ytrain == 1).sum())
    scale_pos_weight = neg / pos
    print(f"      Class balance -> negatives {neg}, positives {pos}")
    print(f"      scale_pos_weight = {scale_pos_weight:.4f}")

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # 4. Define the hyperparameter grid.
    # -------------------------------------------------------------------------
    print("\n[4/7] Defining the hyperparameter grid")
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }
    total_combos = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"      {total_combos} parameter combinations x 5 folds = {total_combos * 5} fits")
    for key, values in param_grid.items():
        print(f"      {key:<28}: {values}")

    # -------------------------------------------------------------------------
    # 5. Tune with stratified 5-fold cross-validation, optimising F1.
    # -------------------------------------------------------------------------
    print("\n[5/7] Running GridSearchCV (scoring = f1)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    with mlflow.start_run(run_name="xgboost_gridsearch") as parent_run:
        search.fit(Xtrain, ytrain)
        print(f"      Best CV F1 score : {search.best_score_:.4f}")
        print(f"      Best parameters  : {search.best_params_}")

        # ---------------------------------------------------------------------
        # 6. Log EVERY tuned combination as a nested MLflow run, so the full
        #    search - not just the winner - is reproducible and comparable.
        # ---------------------------------------------------------------------
        print("\n[6/7] Logging all tuned parameter sets to MLflow")
        results = search.cv_results_
        for idx in range(len(results["params"])):
            with mlflow.start_run(run_name=f"candidate_{idx:03d}", nested=True):
                mlflow.log_params(
                    {k.replace("model__", ""): v for k, v in results["params"][idx].items()}
                )
                mlflow.log_metrics(
                    {
                        "mean_cv_f1": float(results["mean_test_score"][idx]),
                        "std_cv_f1": float(results["std_test_score"][idx]),
                        "mean_train_f1": float(results["mean_train_score"][idx]),
                    }
                )
        print(f"      Logged {len(results['params'])} candidate runs.")

        # ---------------------------------------------------------------------
        # 7. Evaluate the winning model on both splits and log the results.
        # ---------------------------------------------------------------------
        print("\n[7/7] Evaluating the best model")
        best_model = search.best_estimator_
        train_metrics = evaluate(best_model, Xtrain, ytrain, "train")
        test_metrics = evaluate(best_model, Xtest, ytest, "test")

        mlflow.log_params(
            {k.replace("model__", "best_"): v for k, v in search.best_params_.items()}
        )
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics({k: float(v) for k, v in train_metrics.items()})
        mlflow.log_metrics({k: float(v) for k, v in test_metrics.items()})

        # Register the fitted pipeline as an MLflow artifact as well.
        try:
            mlflow.sklearn.log_model(best_model, name="tourism_model")
        except TypeError:
            # Older MLflow releases use artifact_path instead of name.
            mlflow.sklearn.log_model(best_model, artifact_path="tourism_model")
        except Exception as exc:  # noqa: BLE001
            print(f"      MLflow model logging skipped: {exc}")

        print(f"\n      MLflow run id: {parent_run.info.run_id}")

    # -------------------------------------------------------------------------
    # Persist the model into the deployment folder. The workflow commits this
    # file back to main, and Streamlit Community Cloud serves it from there.
    # -------------------------------------------------------------------------
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n      Best model saved -> {MODEL_PATH}")

    summary = {
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "scale_pos_weight": round(scale_pos_weight, 4),
        "best_cv_f1": round(float(search.best_score_), 4),
        **{k: round(float(v), 4) for k, v in train_metrics.items()},
        **{k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"      Metrics summary saved -> {METRICS_PATH}")

    print("\n" + "=" * 78)
    print("MODEL BUILDING COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
