# hack-nu-26

Churn prediction pipeline: a **two-stage hierarchical** classifier that predicts whether a user churns, and (among churned users) whether churn is **voluntary** or **involuntary**. Training, time-series cross-validation, and test inference live in **`training_pipeline.ipynb`**.

## Model overview

1. **Stage 1** — binary: stayed (`not_churned`) vs churned.
2. **Stage 2** — binary on churned users only: `vol_churn` vs `invol_churn`.

Final three-class probabilities are combined hierarchically:

- \(P(\text{not}) = p_{\text{stay}}\)
- \(P(\text{vol}) = (1 - p_{\text{stay}}) \cdot P(\text{vol} \mid \text{churn})\)
- \(P(\text{invol}) = (1 - p_{\text{stay}}) \cdot (1 - P(\text{vol} \mid \text{churn}))\)

You can pick **different** base learners per stage (`STAGE1_MODEL`, `STAGE2_MODEL`), e.g. CatBoost for stage 1 and a voting ensemble (XGBoost + CatBoost on ordinal-encoded features) for stage 2.

**Temporal ensemble:** predictions blend a model fit on **full** training data with one fit on the **late** slice of the training period (`LATE_FRAC`, `ENSEMBLE_ALPHA`) to reduce drift vs the test-era distribution.

## Repository layout

| Path | Purpose |
|------|---------|
| `training_pipeline.ipynb` | End-to-end feature build, CV, hyperparameter hooks, test inference, `submission.csv` |
| `requirements.txt` | Python dependencies |
| `best_model_params.json` | Cached best hyperparameters per model family (when tuning writes checkpoints) |
| `submission.csv` | Default competition-style output (`user_id`, `churn_status`) |
| `weights/` | Optional exported checkpoints from the **late-data** training pass (`stage1.*`, `stage2.*`) |
| `data/preprocessed/` | Primary inputs used by the notebook (`train/`, `test/`) |
| `data/train/`, `data/test/` | Raw tables paired with preprocessed files for merges / analysis |
| `data/merged/` | Output of the merge script (combined preprocessed + raw columns) |
| `scripts/merge_train_sources.py` | Merge preprocessed and raw CSVs on natural keys |
| `feature_engineering/` | Standalone feature scripts |
| `alt_data/` | Additional feature tables (optional) |

## Setup

```bash
cd hack-nu-26
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies include **NumPy**, **pandas**, **scikit-learn**, **XGBoost**, **CatBoost**, and **SciPy** (see `requirements.txt` for pinned versions).

## Running the pipeline

1. Open **`training_pipeline.ipynb`** in Jupyter, VS Code, or Cursor.
2. Adjust **paths** in the notebook if needed. Some cells use absolute paths (e.g. `DATA_DIR`, `BEST_PARAMS_PATH`, `SUBMISSION_PATH`, `TEST_DIR`); point them at your local clone of this repo.
3. Run cells in order. The last section (**§4**) trains on full data (plus the temporal late slice), runs inference on the test set, and writes **`submission.csv`**.
4. After a successful §4 run, **`weights/stage1`** and **`weights/stage2`** hold one checkpoint per stage (`.cbm` for CatBoost / voting CatBoost leg, `.json` for XGBoost-only stages). Only the **late-data** fit is persisted, to match the recency-weighted ensemble.

## Merging preprocessed and raw tables

```bash
python scripts/merge_train_sources.py --help
```

Defaults in the script point at **`data/preprocessed/test`** + **`data/test`** → **`data/merged/test`**. Override `--preprocessed-dir`, `--raw-dir`, and `--out-dir` for train or custom layouts. Use `--include-generations` if you need the large generations file.

## Configuration highlights (notebook)

- **`STAGE1_MODEL` / `STAGE2_MODEL`**: `"xgb"`, `"catboost"`, or `"voting"` (per-stage).
- **`ENSEMBLE_ALPHA`**: weight on the **full-data** model; `1 - alpha` on the **late** model.
- **`LATE_FRAC`**: fraction of the **start** of the training index range dropped before fitting the late model (e.g. `0.5` keeps the latest half).
- **`SUBMISSION_PATH`**: where to write predictions.

## License / data

Add your team’s license and data-use terms here if applicable.
