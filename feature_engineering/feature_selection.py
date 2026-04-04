#!/usr/bin/env python3
"""
Build the same user-level frame as ``feature_engineering.ipynb``, score numeric
columns vs ``y_churned``, drop weak signals, then prune near-duplicate columns.

Default weak-signal rule: drop if ``mutual_info < 0.015`` **or**
``|Pearson vs y_churned| < 0.07``. Optional gates: ``--min-abs-spearman``,
``--max-f-pvalue``, ``--max-chi2-pvalue`` (defaults disable these).
``subscription_start_ts`` is always kept for time-based CV.

Collinearity: while some pair has ``|r| >= 0.92`` (median-imputed), drop the
member with lower MI (protected columns are never dropped first).

Example::

    /home/ansar/work/.venv/bin/python feature_selection.py

Outputs (``--out-dir``, default ``data/feature_engineering/``): ``thresholds.json``,
``univariate_train_y_churned.csv``, ``dropped_features.csv``,
``feature_manifest.json``, ``train/`` (like ``preprocessed/train/``) and by default
``test/`` (like ``preprocessed/test/``), each with split ``*_users*.csv`` plus row-level
copies unless ``--no-copy-raw``. Optional wide tables (``--wide-csv``):
``train_user_level_selected.csv`` and ``test_user_level_selected.csv``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42

# Matches ``train_users_quizzes.csv`` column order (excluding user_id).
QUIZ_COLUMNS = [
    "source",
    "team_size",
    "experience",
    "usage_plan",
    "frustration",
    "first_feature",
    "role",
]

def _split_prefix(data_dir: Path) -> str:
    n = data_dir.name
    if n not in ("train", "test"):
        raise ValueError(f"Expected preprocessed folder name 'train' or 'test', got {n!r}")
    return n


def _write_indexed_csv(path: Path, frame: pd.DataFrame) -> None:
    """Leading unnamed column + 0..n-1 row labels, like ``preprocessed/train/*.csv``."""
    out = frame.copy()
    out.insert(0, "", range(len(out)))
    out.to_csv(path, index=False)


def write_preprocessed_style_split(
    source_dir: Path,
    out_split: Path,
    df: pd.DataFrame,
    kept_numeric: list[str],
    copy_raw: bool,
    split: str,
) -> dict[str, str]:
    """
    Write ``{train|test}_users*.csv`` split tables and optionally copy row-level CSVs
    from ``source_dir`` (preprocessed train or test folder).
    """
    if split not in ("train", "test"):
        raise ValueError(split)
    p = split
    out_split.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    prop_src = pd.read_csv(
        source_dir / f"{p}_users_properties.csv",
        usecols=["user_id", "subscription_start_date", "subscription_plan", "country_code"],
    )
    prop = prop_src.merge(df[["user_id", "subscription_start_ts"]], on="user_id", how="left")
    _write_indexed_csv(out_split / f"{p}_users_properties.csv", prop)
    written[f"{p}_users_properties.csv"] = str((out_split / f"{p}_users_properties.csv").resolve())

    if split == "train":
        users = df[["user_id", "churn_status"]]
    else:
        users = df[["user_id"]]
    _write_indexed_csv(out_split / f"{p}_users.csv", users)
    written[f"{p}_users.csv"] = str((out_split / f"{p}_users.csv").resolve())

    quiz_cols = [c for c in QUIZ_COLUMNS if c in df.columns]
    quiz = df[["user_id"] + quiz_cols]
    _write_indexed_csv(out_split / f"{p}_users_quizzes.csv", quiz)
    written[f"{p}_users_quizzes.csv"] = str((out_split / f"{p}_users_quizzes.csv").resolve())

    num_rest = [c for c in kept_numeric if c != "subscription_start_ts"]
    if num_rest:
        sel_num = df[["user_id"] + num_rest]
        _write_indexed_csv(out_split / f"{p}_users_selected_numerics.csv", sel_num)
        written[f"{p}_users_selected_numerics.csv"] = str(
            (out_split / f"{p}_users_selected_numerics.csv").resolve()
        )

    if copy_raw:
        for suffix in ("purchases", "transaction_attempts", "generations"):
            name = f"{p}_users_{suffix}.csv"
            src = source_dir / name
            dst = out_split / name
            shutil.copy2(src, dst)
            written[name] = str(dst.resolve())

    return written


def load_base_users(data_dir: Path) -> pd.DataFrame:
    sp = _split_prefix(data_dir)
    if sp == "train":
        u = pd.read_csv(data_dir / "train_users.csv", usecols=["user_id", "churn_status"])
    else:
        u = pd.read_csv(data_dir / "test_users.csv", usecols=["user_id"])
    prop = pd.read_csv(
        data_dir / f"{sp}_users_properties.csv",
        usecols=["user_id", "subscription_start_date", "subscription_plan", "country_code"],
    )
    q = pd.read_csv(
        data_dir / f"{sp}_users_quizzes.csv",
        usecols=[
            "user_id",
            "source",
            "team_size",
            "experience",
            "usage_plan",
            "frustration",
            "first_feature",
            "role",
        ],
    )
    df = u.merge(prop, on="user_id", how="left").merge(q, on="user_id", how="left")
    df["subscription_start_ts"] = pd.to_datetime(
        df["subscription_start_date"], utc=True, errors="coerce"
    ).astype("int64")
    return df.drop(columns=["subscription_start_date"])


def aggregate_purchases_attempts(data_dir: Path) -> pd.DataFrame:
    sp = _split_prefix(data_dir)
    pur = pd.read_csv(data_dir / f"{sp}_users_purchases.csv", low_memory=False)
    ta = pd.read_csv(data_dir / f"{sp}_users_transaction_attempts.csv", low_memory=False)
    for c in ta.columns:
        if c.startswith("is_") or c in {"is_3d_secure", "is_3d_secure_authenticated"}:
            ta[c] = ta[c].map(lambda x: str(x).lower() in {"true", "1"})
    m = pur.merge(ta, on="transaction_id", how="left", suffixes=("_pur", ""))
    bool_cols = [c for c in ta.columns if c.startswith("is_")]
    agg_dict = {
        "transaction_id": "count",
        "purchase_amount_dollars": "sum",
        "purchase_type": "nunique",
        "amount_in_usd": "sum",
    }
    for bc in bool_cols:
        if bc in m.columns:
            agg_dict[bc] = "mean"
    g = m.groupby("user_id", as_index=False).agg(agg_dict)
    rename = {
        "transaction_id": "purch_n",
        "purchase_amount_dollars": "purch_amount_sum",
        "purchase_type": "purch_type_nunique",
        "amount_in_usd": "att_amount_sum",
    }
    g = g.rename(columns=rename)
    for bc in bool_cols:
        if bc in g.columns:
            g = g.rename(columns={bc: f"att_mean_{bc}"})
    return g


def aggregate_generations(data_dir: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    sp = _split_prefix(data_dir)
    path = data_dir / f"{sp}_users_generations.csv"
    usecols = ["user_id", "status", "generation_type"]
    status_parts: list[pd.Series] = []
    type_parts: list[pd.Series] = []
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        status_parts.append(chunk.groupby(["user_id", "status"]).size())
        type_parts.append(chunk.groupby(["user_id", "generation_type"]).size())
    st = pd.concat(status_parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    st.columns = [f"gen_status_{c}" for c in st.columns.astype(str)]
    gt = pd.concat(type_parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    gt.columns = [f"gen_type_{c}" for c in gt.columns.astype(str)]
    out = st.join(gt, how="outer").fillna(0).astype(np.float32)
    out["gen_total"] = out[[c for c in out.columns if c.startswith("gen_status_")]].sum(axis=1)
    return out.reset_index()


def add_engineered_features(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    eps = 1e-6
    if {"purch_amount_sum", "purch_n"}.issubset(out.columns):
        out["fe_avg_purchase_amount"] = out["purch_amount_sum"] / (out["purch_n"] + eps)
    if {"att_amount_sum", "purch_amount_sum"}.issubset(out.columns):
        out["fe_att_to_purch_amount_ratio"] = out["att_amount_sum"] / (
            out["purch_amount_sum"].abs() + eps
        )
    if {"gen_total", "purch_n"}.issubset(out.columns):
        out["fe_gen_per_purchase"] = out["gen_total"] / (out["purch_n"] + eps)
    for col in ["purch_amount_sum", "att_amount_sum", "gen_total", "purch_n"]:
        if col in out.columns:
            out[f"fe_log1p_{col}"] = np.log1p(out[col].clip(lower=0).fillna(0))
    completed = [
        c for c in out.columns if c == "gen_status_completed" or str(c).endswith("_completed")
    ]
    if "gen_total" in out.columns and completed:
        num_completed = out[completed].sum(axis=1)
        out["fe_gen_completed_rate"] = num_completed / (out["gen_total"] + eps)
    att_mean_cols = [c for c in out.columns if c.startswith("att_mean_is_")]
    if att_mean_cols:
        out["fe_att_3ds_rate"] = out.get("att_mean_is_3d_secure", pd.Series(np.nan, index=out.index))
        if "att_mean_is_3d_secure_authenticated" in out.columns:
            out["fe_att_3ds_auth_given_3ds"] = out["att_mean_is_3d_secure_authenticated"] / (
                out["att_mean_is_3d_secure"] + eps
            )
    return out


def build_frame(data_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = load_base_users(data_dir)
    df = df.merge(aggregate_purchases_attempts(data_dir), on="user_id", how="left")
    df = df.merge(aggregate_generations(data_dir), on="user_id", how="left")
    df = add_engineered_features(df)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_feat = {"user_id", "churn_status"}
    cat_cols = [c for c in df.columns if c not in num_cols and c not in non_feat]
    for c in cat_cols:
        df[c] = df[c].fillna("skipped").astype(str)
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df = df.sort_values("subscription_start_ts").reset_index(drop=True)
    return df, num_cols, cat_cols


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out["churn_status"]
    out["y_churned"] = (s != "not_churned").astype(np.float64)
    out["y_vol_if_churned"] = np.where(
        s != "not_churned", (s == "vol_churn").astype(np.float64), np.nan
    )
    out["y_churn_ordinal"] = s.map({"not_churned": 0.0, "vol_churn": 1.0, "invol_churn": 2.0}).astype(
        np.float64
    )
    return out


def numeric_feature_names(df: pd.DataFrame) -> list[str]:
    analysis_exclude = {"user_id", "churn_status", "y_churned", "y_churn_ordinal", "y_vol_if_churned"}
    return [
        c
        for c in df.columns
        if c not in analysis_exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def univariate_table(
    frame: pd.DataFrame, feature_cols: list[str], y: pd.Series
) -> pd.DataFrame:
    X = frame[feature_cols].to_numpy(dtype=np.float64)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    y_arr = y.to_numpy().astype(np.int64)

    mi = mutual_info_classif(X_imp, y_arr, random_state=RANDOM_STATE, n_neighbors=3)
    _, f_pvals = f_classif(X_imp, y_arr)
    X_nn = X_imp - X_imp.min(axis=0)
    _, chi_pvals = chi2(X_nn, y_arr)

    ys = pd.Series(y_arr.astype(float), index=frame.index)
    pear = frame[feature_cols].corrwith(ys, method="pearson")
    spear = frame[feature_cols].corrwith(ys, method="spearman")

    return pd.DataFrame(
        {
            "feature": feature_cols,
            "mutual_info": mi,
            "abs_pearson": pear.abs().values,
            "abs_spearman": spear.abs().values,
            "f_pvalue": f_pvals,
            "chi2_pvalue": chi_pvals,
        }
    )


def weak_signal_mask(
    uni: pd.DataFrame,
    min_mi: float,
    min_abs_pearson: float,
    min_abs_spearman: float,
    max_f_pvalue: float,
    max_chi2_pvalue: float,
    protected: frozenset[str],
) -> pd.Series:
    """True = drop (fails at least one active threshold). Protected columns never drop here."""
    m = pd.Series(False, index=uni.index)
    if min_mi > 0:
        m |= uni["mutual_info"] < min_mi
    if min_abs_pearson > 0:
        m |= uni["abs_pearson"] < min_abs_pearson
    if min_abs_spearman > 0:
        m |= uni["abs_spearman"] < min_abs_spearman
    if max_f_pvalue < 1.0:
        m |= uni["f_pvalue"] > max_f_pvalue
    if max_chi2_pvalue < 1.0:
        m |= uni["chi2_pvalue"].fillna(1.0) > max_chi2_pvalue
    for f in protected:
        if f in m.index:
            m.loc[f] = False
    return m


def align_to_selected_features(
    df: pd.DataFrame, kept_numeric: list[str], cat_cols: list[str]
) -> pd.DataFrame:
    """Ensure train-selected columns exist (e.g. test may lack some ``gen_type_*`` levels)."""
    out = df.copy()
    for c in kept_numeric:
        if c not in out.columns:
            out[c] = np.nan
    for c in cat_cols:
        if c not in out.columns:
            out[c] = "skipped"
    return out


def correlation_matrix_imputed(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = frame[cols].to_numpy(dtype=np.float64)
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    d = pd.DataFrame(X_imp, columns=cols, index=frame.index)
    return d.corr(numeric_only=True)


def prune_collinear(
    survivors: list[str],
    corr: pd.DataFrame,
    mi_by_feat: dict[str, float],
    abs_r_min: float,
    protected: frozenset[str],
) -> tuple[list[str], list[dict[str, str | float]]]:
    """Iteratively remove one feature from the pair with |r| >= abs_r_min (keep higher MI)."""
    keep = set(survivors)
    dropped: list[dict[str, str | float]] = []
    while True:
        worst_pair: tuple[str, str] | None = None
        worst_r = abs_r_min
        kl = sorted(keep)
        for i, a in enumerate(kl):
            for b in kl[i + 1 :]:
                r = abs(float(corr.loc[a, b]))
                if r >= abs_r_min - 1e-12 and r >= worst_r - 1e-12:
                    if worst_pair is None or r > worst_r + 1e-12:
                        worst_r = r
                        worst_pair = (a, b)
        if worst_pair is None:
            break
        a, b = worst_pair
        if a in protected and b in protected:
            break
        if a in protected:
            keep.remove(b)
            dropped.append(
                {
                    "feature": b,
                    "reason": "collinear",
                    "kept_correlated_with": a,
                    "abs_pearson": float(corr.loc[a, b]),
                }
            )
            continue
        if b in protected:
            keep.remove(a)
            dropped.append(
                {
                    "feature": a,
                    "reason": "collinear",
                    "kept_correlated_with": b,
                    "abs_pearson": float(corr.loc[a, b]),
                }
            )
            continue
        ma, mb = mi_by_feat.get(a, 0.0), mi_by_feat.get(b, 0.0)
        if ma >= mb:
            drop, stay = b, a
        else:
            drop, stay = a, b
        keep.remove(drop)
        dropped.append(
            {
                "feature": drop,
                "reason": "collinear",
                "kept_correlated_with": stay,
                "abs_pearson": float(corr.loc[a, b]),
            }
        )
    return sorted(keep), dropped


def run(
    train_dir: Path,
    test_dir: Path,
    out_dir: Path,
    min_mi: float,
    min_abs_pearson: float,
    min_abs_spearman: float,
    max_f_pvalue: float,
    max_chi2_pvalue: float,
    collinear_abs_r: float,
    write_train_tables: bool,
    write_test_tables: bool,
    copy_raw_tables: bool,
    write_wide_csv: bool,
) -> None:
    np.random.seed(RANDOM_STATE)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _num, cat_cols = build_frame(train_dir)
    df = add_targets(df)
    feats = numeric_feature_names(df)

    uni = univariate_table(df, feats, df["y_churned"])
    uni = uni.set_index("feature")

    protected = frozenset({"subscription_start_ts"})
    bad = weak_signal_mask(
        uni,
        min_mi=min_mi,
        min_abs_pearson=min_abs_pearson,
        min_abs_spearman=min_abs_spearman,
        max_f_pvalue=max_f_pvalue,
        max_chi2_pvalue=max_chi2_pvalue,
        protected=protected,
    )
    weak_dropped = uni.index[bad.values].tolist()
    survivors = uni.index[~bad.values].tolist()

    weak_records = []
    for f in weak_dropped:
        row = uni.loc[f]
        reasons = []
        if min_mi > 0 and row["mutual_info"] < min_mi:
            reasons.append(f"mi<{min_mi}")
        if min_abs_pearson > 0 and row["abs_pearson"] < min_abs_pearson:
            reasons.append(f"|pearson|<{min_abs_pearson}")
        if min_abs_spearman > 0 and row["abs_spearman"] < min_abs_spearman:
            reasons.append(f"|spearman|<{min_abs_spearman}")
        if max_f_pvalue < 1.0 and row["f_pvalue"] > max_f_pvalue:
            reasons.append(f"f_pvalue>{max_f_pvalue}")
        if max_chi2_pvalue < 1.0 and float(row["chi2_pvalue"]) > max_chi2_pvalue:
            reasons.append(f"chi2_pvalue>{max_chi2_pvalue}")
        weak_records.append({"feature": f, "reason": "weak_signal", "detail": "; ".join(reasons)})

    corr_full = correlation_matrix_imputed(df, survivors)
    mi_by_feat = uni["mutual_info"].to_dict()
    kept_numeric, collinear_dropped = prune_collinear(
        survivors, corr_full, mi_by_feat, collinear_abs_r, protected
    )

    thresholds = {
        "min_mutual_info": min_mi,
        "min_abs_pearson": min_abs_pearson,
        "min_abs_spearman": min_abs_spearman,
        "max_f_pvalue": max_f_pvalue,
        "max_chi2_pvalue": max_chi2_pvalue,
        "collinear_abs_pearson": collinear_abs_r,
        "protected_numeric": sorted(protected),
        "target_for_univariate": "y_churned",
    }
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    univariate_out = uni.reset_index()
    univariate_out.to_csv(out_dir / "univariate_train_y_churned.csv", index=False)

    dropped_rows = weak_records + collinear_dropped
    pd.DataFrame(dropped_rows).to_csv(out_dir / "dropped_features.csv", index=False)

    manifest: dict = {
        "train_dir": str(train_dir.resolve()),
        "n_rows": int(len(df)),
        "numeric_input": len(feats),
        "numeric_after_weak_filter": len(survivors),
        "numeric_final": len(kept_numeric),
        "categorical_columns": cat_cols,
        "kept_numeric": kept_numeric,
        "protected_numeric_always_kept": sorted(protected),
    }

    cols_out = ["user_id", "churn_status"] + kept_numeric + cat_cols
    df_out = df[cols_out]

    manifest["output_train_dir"] = None
    manifest["train_preprocessed_style_files"] = None
    manifest["test_dir"] = str(test_dir.resolve())
    manifest["n_test_rows"] = None
    manifest["output_test_dir"] = None
    manifest["test_preprocessed_style_files"] = None
    manifest["output_wide_test_csv"] = None

    if write_train_tables:
        train_out = out_dir / "train"
        pre_train = write_preprocessed_style_split(
            train_dir, train_out, df, kept_numeric, copy_raw=copy_raw_tables, split="train"
        )
        manifest["output_train_dir"] = str(train_out.resolve())
        manifest["train_preprocessed_style_files"] = pre_train

    df_test, _nt, _ct = build_frame(test_dir)
    df_test = align_to_selected_features(df_test, kept_numeric, cat_cols)
    manifest["n_test_rows"] = int(len(df_test))

    if write_test_tables:
        test_out = out_dir / "test"
        pre_test = write_preprocessed_style_split(
            test_dir, test_out, df_test, kept_numeric, copy_raw=copy_raw_tables, split="test"
        )
        manifest["output_test_dir"] = str(test_out.resolve())
        manifest["test_preprocessed_style_files"] = pre_test

    if write_wide_csv:
        csv_path = out_dir / "train_user_level_selected.csv"
        df_out.to_csv(csv_path, index=False)
        manifest["output_wide_csv"] = str(csv_path.resolve())
        test_wide = out_dir / "test_user_level_selected.csv"
        df_test_wide = df_test[["user_id"] + kept_numeric + cat_cols]
        df_test_wide.to_csv(test_wide, index=False)
        manifest["output_wide_test_csv"] = str(test_wide.resolve())

    (out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    parts = ["manifest", "thresholds", "univariate", "dropped_features"]
    if write_train_tables:
        parts.append(f"train/ ({len(manifest.get('train_preprocessed_style_files') or {})} files)")
    if write_test_tables:
        parts.append(f"test/ ({len(manifest.get('test_preprocessed_style_files') or {})} files)")
    if write_wide_csv:
        parts.append("train_user_level_selected.csv + test_user_level_selected.csv")
    print(f"Wrote {out_dir}/ ({', '.join(parts)})")


def main() -> None:
    p = argparse.ArgumentParser(description="Select numeric features and write data/feature_engineering artifacts.")
    p.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/home/ansar/work/hack-nu-26/data/preprocessed/train"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/ansar/work/hack-nu-26/data/feature_engineering"),
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Preprocessed test folder (default: parent of --train-dir / test).",
    )
    p.add_argument("--min-mi", type=float, default=0.015)
    p.add_argument("--min-abs-pearson", type=float, default=0.07)
    p.add_argument(
        "--min-abs-spearman",
        type=float,
        default=0.0,
        help="If >0, drop when |Spearman| below this. Default 0 = off.",
    )
    p.add_argument(
        "--max-f-pvalue",
        type=float,
        default=1.0,
        help="If <1, drop when ANOVA F p-value exceeds this. Default 1 = off.",
    )
    p.add_argument(
        "--max-chi2-pvalue",
        type=float,
        default=1.0,
        help="If <1, drop when chi2 p-value exceeds this. Default 1 = off.",
    )
    p.add_argument("--collinear-abs-r", type=float, default=0.92)
    p.add_argument(
        "--no-train-tables",
        action="store_true",
        help="Do not write ``train/`` with preprocessed-style split CSVs.",
    )
    p.add_argument(
        "--no-test-tables",
        action="store_true",
        help="Do not write ``test/`` (default is to write it like ``train/``).",
    )
    p.add_argument(
        "--no-copy-raw",
        action="store_true",
        help="When writing ``train/``, do not copy purchases / attempts / generations CSVs.",
    )
    p.add_argument(
        "--wide-csv",
        action="store_true",
        help="Also write ``train_user_level_selected.csv`` (single wide table).",
    )
    args = p.parse_args()
    test_dir = args.test_dir if args.test_dir is not None else args.train_dir.parent / "test"
    run(
        train_dir=args.train_dir,
        test_dir=test_dir,
        out_dir=args.out_dir,
        min_mi=args.min_mi,
        min_abs_pearson=args.min_abs_pearson,
        min_abs_spearman=args.min_abs_spearman,
        max_f_pvalue=args.max_f_pvalue,
        max_chi2_pvalue=args.max_chi2_pvalue,
        collinear_abs_r=args.collinear_abs_r,
        write_train_tables=not args.no_train_tables,
        write_test_tables=not args.no_test_tables,
        copy_raw_tables=not args.no_copy_raw,
        write_wide_csv=args.wide_csv,
    )


if __name__ == "__main__":
    main()
