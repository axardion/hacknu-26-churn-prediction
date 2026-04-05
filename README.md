# hack-nu-26 — script reference

This README lists **what each script in the repo does**. Paths are relative to the repository root unless noted.

---

## Root directory

| Script | What it does |
|--------|----------------|
| **`preprocess_data.py`** | CLI to preprocess raw train/test CSVs: normalize quiz fields, drop `flow_type`, replace empty strings with `skipped`, filter train rows by country coverage vs test and by `EXCLUDED_TRAIN_COUNTRY_CODES`, etc. Run with `-h` for arguments. |
| **`merge_data_sources.py`** | For `train` and/or `test`, outer-joins each raw CSV under `data/<split>/` with its matching file under `data/preprocessed/<split>/` on natural keys, suffixes `_raw` / `_prep`, writes full merged tables to `data/merged/<split>/`. `--data-root` defaults to `./data`. |
| **`purchases.py`** | Reads `train_users_purchases.csv` / `test_users_purchases.csv`, shifts purchase timestamps, aggregates per user (totals, per purchase-type spend/counts, min/max/mean spend, span of days). Writes `data/purchases_train.csv` and `data/purchases_test.csv` with one row per user from the respective user lists. |
| **`durations.py`** | Processes a generations CSV: per-user completion durations (`completed_at - created_at`), split by image/video/overall; writes derived duration stats to an output CSV (used for duration features). |
| **`countries_files.py`** | Uses `pycountry` and the World Bank API to fetch country indicators and build/update `data/countries.csv` (GDP, growth, population, etc.) for codes seen in properties. Network-heavy; configurable years and indicator set. |
| **`subscription_type.py`** | Joins properties with `data/purchases_train.csv`, computes mean/std of `total_spend` per `subscription_plan`, maps back to all train+test users, saves `data/subscriptions.csv`. |
| **`user_market_anchor_features.py`** | Builds **user-level market anchor features** from external CSVs (social mentions, Google Trends, optional GitHub/HF) aligned to decoded subscription start dates. Documents date shifting (`SHIFT_YEARS`, etc.) and how `is_migration_signal` is defined. Depends on `churn_features` and various `data/*.csv` inputs. |
| **`analyze_model_usage_churn.py`** | Streams `train_users_generations.csv` in chunks, pivots counts by `generation_type`, computes Pearson correlation and mutual information vs binary churn. CLI can override paths. Useful for exploratory analysis of model-usage vs churn. |
| **`_run_experiments.py`** | Standalone experiment runner: mirrors parts of the training notebook (loads data, `TimeSeriesSplit`, hierarchical models). Uses `best_model_params.json`. Intended for quick scripted experiments rather than the main notebook. |
| **`_run_lastfold.py`** | Same modeling idea as `_run_experiments.py` but evaluates **only the last fold** of time-series CV (fold 4 of 5). Faster sanity check on the most recent validation slice. |

---

## `scripts/`

| Script | What it does |
|--------|----------------|
| **`merge_train_sources.py`** | Merges **preprocessed** and **raw** tables (defaults: `data/preprocessed/test` + `data/test` → `data/merged/test`). Per-table outer join on configured keys; overlapping columns get `_prep` / `_raw` suffixes. Writes **head(10)** per file plus a **`combined.csv`** keyed by `user_id`. Override dirs with `--preprocessed-dir`, `--raw-dir`, `--output-dir`; `--only` to limit files; `--include-generations` for the large generations file. |

---

## `feature_engineering/`

### Core modules

| File | What it does |
|------|----------------|
| **`churn_analysis.py`** | Large module: builds a flat per-user feature matrix from Higgsfield-style tables, including **`inter_*`** payment/geo interaction features. Exposes `build_inter_features_without_generations()` (no generations file) and full `build_features(..., generations_path=...)`. Imported by other scripts and notebooks. |
| **`add_features_generations_common.py`** | Shared utilities for generation-based features: chunked CSV reads, merge helpers, CLI (`parse_io_args`, `run_inplace_update`), and `main_for_feature(name)` for simple column appenders. |
| **`feature_selection.py`** | Scores numeric columns vs churn, drops weak/univariate signals and highly collinear columns, writes manifests and optional filtered `train/` / `test/` trees under `data/feature_engineering/` (see file docstring). Many CLI flags. |

### One-off or interaction appenders

| File | What it does |
|------|----------------|
| **`feature_churn_differentiation.py`** | Appends **`inter_payment_risk_score`**, **`inter_mismatch`**, **`inter_country_fail_rate`**, **`inter_3ds_x_geohigh`**, **`inter_debit_x_geohigh`** via `build_inter_features_without_generations`. |
| **`feature_ix_mig_share_zgen.py`** | Merges migration-related columns from alt CSVs with z-scored share × gen-delta features (`ix_zshare_x_zgen_delta`, `mig_any_b5_a5`) into user tables. |

### Generation-derived columns (mostly same CLI pattern)

These read `train_users_generations.csv` (or paths from CLI) and update **`data/train/train_users.csv`** and **`data/test/test_users.csv`** (or `preprocessed` paths—see each script / `add_features_generations_common`):

| File | Column / behavior |
|------|-------------------|
| **`feature_share_video_model_7_times_log1p_gen_total.py`** | `share_video_model_7_times_log1p_gen_total` |
| **`feature_has_any_generation.py`** | `has_any_generation` |
| **`feature_video_gen_share.py`** | `video_gen_share` |
| **`feature_log1p_total_gen.py`** | `log1p_total_gen` |
| **`feature_total_generations.py`** | `total_generations` |
| **`feature_nsfw_rate.py`** | `nsfw_rate` |
| **`feature_success_ratio.py`** | `success_ratio` |
| **`feature_gen_delta_day1_minus_day14.py`** | `gen_delta_day1_minus_day14` |
| **`feature_gen_duration_mean_video.py`** / **`feature_gen_duration_median_video.py`** | Mean/median completion duration for `video_*` generations |
| **`feature_gen_duration_mean_image.py`** / **`feature_gen_duration_median_image.py`** | Same for `image_*` generations |

### Country / economy appenders (read `data/countries.csv` + properties)

Small CSV-based scripts that join **`country_code`** from properties onto **`data/preprocessed/.../train_users.csv`** and **`test_users.csv`**:

| File | Adds (conceptually) |
|------|---------------------|
| **`append_gdp_per_capita.py`** | GDP per capita from country table |
| **`append_gdp_growth_pct.py`** | GDP growth % |
| **`append_prosperity_penetration.py`** | `gdp_per_tiktok_user_scaled` (uses TikTok users + population from country file) |
| **`append_tiktok_penetration.py`** | `tiktok_penetration` |
| **`append_log_penetration.py`** | `log_churn_density` from country churn stats |
| **`append_single_highest_by_gdp_per_capita.py`** | Interaction feature: highest single spend scaled vs GDP per capita |

---

## `test_src/` (ad-hoc analysis and experiments)

These are **not** part of the main training pipeline; they were used for date-shift calibration, plotting, and feature screening.

| File | What it does |
|------|----------------|
| **`revenue.py`** | Loads purchases, applies configurable **date shifts** (days/months/years), aggregates revenue over time, writes plots under `output/revenue_by_*`. |
| **`range.py`** | Prints shifted **min/max** of `purchase_time` on train purchases (sanity check for calendar alignment). |
| **`first49.py`** | Finds first and count of **$49** payments after the same shift, per split. |
| **`experiment.py`** | Large plotting/analysis script: handcrafted formulas on purchase columns, country features, heatmaps, etc. |
| **`experiment_v2.py`** | Churn feature screening on the **last fraction** of users by subscription time; Pearson + mutual information; optional formula columns. |
| **`temporary.py`** | One-off stream: clears `generation_id` cells in `train_users_generations.csv` in place (use with care). |

---

## Notebooks (not `.py`, but primary entrypoints)

| Notebook | Role |
|----------|------|
| **`training_pipeline.ipynb`** | End-to-end: load preprocessed data, engineer features, time-series CV, optional tuning, **full + late** temporal ensemble, test inference, **`submission.csv`**, optional **`weights/stage1.*`** and **`weights/stage2.*`**. This is the main training and submission entry. |

---

## Dependencies

See **`requirements.txt`** for pinned packages (pandas, scikit-learn, XGBoost, CatBoost, etc.). Create a venv and run `pip install -r requirements.txt` before executing scripts that import ML libraries.
