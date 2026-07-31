# Visit with Us — Wellness Tourism Package Prediction (MLOps)

End-to-end MLOps pipeline that predicts whether a customer will purchase the
newly introduced **Wellness Tourism Package**, so the sales team can target the
right customers *before* contacting them.

The whole workflow — data registration, preparation, training with experiment
tracking, and deployment — runs automatically on **GitHub Actions**, and the app
is served from **Streamlit Community Cloud**.

---

## Repository structure

```
.
├── .github/
│   └── workflows/
│       └── pipeline.yml                  # CI/CD workflow (4 jobs)
├── tourism_project/
│   ├── data/
│   │   └── tourism.csv                   # registered raw dataset
│   ├── model_building/
│   │   ├── data_register.py              # schema validation + summary report
│   │   ├── prep.py                       # cleaning + stratified train/test split
│   │   └── train.py                      # tuning, MLflow tracking, evaluation
│   ├── deployment/
│   │   ├── app.py                        # Streamlit front end
│   │   ├── requirements.txt              # serving dependencies
│   │   ├── best_tourism_model_v1.joblib  # committed by the pipeline
│   │   └── model_metrics.json            # committed by the pipeline
│   └── requirements.txt                  # pipeline dependencies
└── README.md
```

---

## Pipeline

| # | Job | What it does | Hand-off |
|---|-----|--------------|----------|
| 1 | `register-dataset` | Checks the CSV exists, validates every expected column, prints a full data summary. Fails the run early if the data contract breaks. | `registered-data` artifact |
| 2 | `data-prep` | Drops `Unnamed: 0` / `CustomerID`, fixes the `Fe Male` typo, removes duplicates, stratified 80/20 split. | `data-splits` artifact |
| 3 | `model-training` | Loads the artifact, tunes an XGBoost pipeline with `GridSearchCV`, logs every candidate to MLflow, evaluates, saves the best model — then **commits it back to `main`**. | commit to `main` |
| 4 | `deployment-check` | Re-checks out `main` and verifies the app, requirements and model artifact are all present. | — |

The training job commits with `[skip ci]` so the automated push does not
re-trigger the workflow.

---

## Model

A single scikit-learn `Pipeline` containing **all** preprocessing plus the
estimator, so the serving app can pass in a raw DataFrame:

- Numeric: median imputation → `StandardScaler`
- Categorical: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`
- Estimator: `XGBClassifier` with `scale_pos_weight` for the ~19 % positive rate
- Tuned with 5-fold stratified `GridSearchCV`, **scored on F1** rather than
  accuracy (a "nobody buys" model would score ~81 % accuracy and be useless)

---

## Running it

**Automatically** — push to `main`, or open the *Actions* tab and run the
workflow manually.

**Locally**

```bash
pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
streamlit run tourism_project/deployment/app.py
```

---

## Deployment

Deployed on Streamlit Community Cloud from this repository:

- **Main file path:** `tourism_project/deployment/app.py`
- **Branch:** `main`
- **Python version:** 3.11 (set under *Advanced settings*, so the serving
  environment matches the training environment)

Every pipeline run commits a fresh model, and Streamlit redeploys on the new
commit — continuous delivery with no manual step.
