#!/usr/bin/env python3
"""
Analyze how per-user generation model usage relates to churn vs retention.

- Churn = vol_churn OR invol_churn; retention = all other churn_status values.
- Aggregates train_users_generations.csv in chunks: counts per user × generation_type.
- Importance (stage 1): **Pearson correlation** with binary churn (point-biserial) and **mutual information**.

Positive Pearson → higher feature values associate with **more churn** (less retention).
Negative Pearson → associate with **more retention**.

Usage:
  python analyze_model_usage_churn.py
  python analyze_model_usage_churn.py --train-users data/train/train_users.csv --gens data/train/train_users_generations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def read_users_churn(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    c0 = df.columns[0]
    if str(c0).startswith("Unnamed") or str(c0).lower() == "skipped":
        df = df.drop(columns=c0)
    df["churn"] = df["churn_status"].isin(["vol_churn", "invol_churn"]).astype(np.int8)
    return df[["user_id", "churn"]]


def aggregate_user_model_counts(gens_path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    n_chunks = 0
    for ch in pd.read_csv(
        gens_path,
        chunksize=chunksize,
        usecols=["user_id", "generation_type"],
        low_memory=False,
        dtype={"user_id": str, "generation_type": str},
    ):
        n_chunks += 1
        ch["generation_type"] = ch["generation_type"].fillna("__NA__")
        g = ch.groupby(["user_id", "generation_type"], observed=False).size().reset_index(name="n")
        parts.append(g)
        if n_chunks % 20 == 0:
            print("  chunks", n_chunks, flush=True)
    print("  concat + pivot...", flush=True)
    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.groupby(["user_id", "generation_type"], observed=False)["n"].sum().reset_index()
    wide = long_df.pivot(index="user_id", columns="generation_type", values="n").fillna(0.0)
    wide = wide.rename(columns={c: f"gen_cnt__{c}" for c in wide.columns})
    wide["gen_total"] = wide.sum(axis=1)
    return wide.reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-users", type=Path, default=Path("data/train/train_users.csv"))
    p.add_argument("--gens", type=Path, default=Path("data/train/train_users_generations.csv"))
    p.add_argument("--chunksize", type=int, default=500_000)
    args = p.parse_args()

    print("Load users + churn label (vol + invol = churn)...")
    users = read_users_churn(args.train_users)
    print("  n users:", len(users))
    print("  churn rate:", float(users["churn"].mean()))

    print("Aggregate generations (chunked)...")
    gen_wide = aggregate_user_model_counts(args.gens, chunksize=args.chunksize)
    print("  users with >=1 generation row:", len(gen_wide))

    df = users.merge(gen_wide, on="user_id", how="left")
    count_cols = [c for c in df.columns if c.startswith("gen_cnt__")]
    df[count_cols] = df[count_cols].fillna(0.0)
    df["gen_total"] = df["gen_total"].fillna(0.0)

    denom = df["gen_total"].replace(0, np.nan)
    prop_cols = []
    for c in count_cols:
        pname = c.replace("gen_cnt__", "gen_prop__")
        df[pname] = (df[c] / denom).fillna(0.0)
        prop_cols.append(pname)

    y = df["churn"].astype(float)

    # log1p(counts): stabilizes scale for corr / MI
    log_count_names = count_cols + ["gen_total"]
    log_df = pd.DataFrame({c: np.log1p(df[c].astype(np.float64)) for c in log_count_names})
    X_log = log_df.to_numpy(dtype=np.float64)
    X_prop = df[prop_cols].to_numpy(dtype=np.float64)

    # Pearson with binary churn = point-biserial correlation
    pearson_log = log_df.corrwith(y)
    prop_only = df[prop_cols].astype(np.float64)
    pearson_prop = prop_only.corrwith(y)

    mi_log = mutual_info_classif(X_log, df["churn"].to_numpy(), discrete_features=False, random_state=0)
    mi_prop = mutual_info_classif(X_prop, df["churn"].to_numpy(), discrete_features=False, random_state=0)

    def short_name(col: str) -> str:
        return col.replace("gen_cnt__", "").replace("gen_prop__", "")

    out_log = pd.DataFrame(
        {
            "model_or_total": [short_name(c) for c in log_count_names],
            "pearson_vs_churn_log1p": pearson_log.values,
            "abs_pearson": np.abs(pearson_log.values),
            "mi_log1p": mi_log,
        }
    ).sort_values("abs_pearson", ascending=False)

    out_prop = pd.DataFrame(
        {
            "model": [short_name(c) for c in prop_cols],
            "pearson_vs_churn_share": pearson_prop.values,
            "abs_pearson": np.abs(pearson_prop.values),
            "mi_share": mi_prop,
        }
    ).sort_values("abs_pearson", ascending=False)

    print("\n=== Pearson (point-biserial) & MI on log1p(counts) + log1p(gen_total) ===")
    print("Positive Pearson: more generations / total aligns with higher churn (retention lower).")
    print(out_log.to_string(index=False))

    print("\n=== Pearson & MI on within-user share of each model (sums to 1 per user) ===")
    print(out_prop.to_string(index=False))

    z = df["gen_total"] == 0
    if z.any():
        print("\n--- Sanity: users with zero generation rows ---")
        print("  count:", int(z.sum()))
        print("  churn rate when gen_total==0:", round(float(df.loc[z, "churn"].mean()), 4))
        print("  churn rate when gen_total>0 :", round(float(df.loc[~z, "churn"].mean()), 4))


if __name__ == "__main__":
    main()
