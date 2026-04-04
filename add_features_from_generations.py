#!/usr/bin/env python3
"""
Chunk-read generations CSVs and add per-user features (same logic as churn_importance_by_time_bin).

Reads from data/preprocessed/train|test/:
  - train_users.csv, train_users_properties.csv, train_users_generations.csv
  - test_users.csv, test_users_properties.csv, test_users_generations.csv

Writes merged user tables with generation features (default: data/preprocessed/cache/).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.parser import isoparse

CHUNK = 300_000


def read_csv_drop_index(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, **kwargs)
    c0 = df.columns[0]
    if str(c0).startswith("Unnamed") or str(c0).lower() == "skipped":
        df = df.drop(columns=c0)
    return df


def parse_subscription_ts(s: str) -> datetime:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", str(s))
    if not m:
        raise ValueError(f"Bad subscription timestamp: {s!r}")
    y, mo, d, h, mi, se = map(int, m.groups())
    return datetime(y, mo, d, h, mi, se)


def days_from_subscription(created_list, sub_list):
    out = np.empty(len(created_list), dtype=np.float64)
    for i, (c, s) in enumerate(zip(created_list, sub_list)):
        if pd.isna(c) or s is None:
            out[i] = np.nan
            continue
        if getattr(c, "tzinfo", None):
            c = c.replace(tzinfo=None)
        if getattr(s, "tzinfo", None):
            s = s.replace(tzinfo=None)
        out[i] = (c - s).total_seconds() / 86400.0
    return out


def aggregate_generations(base: pd.DataFrame, gen_path: Path) -> pd.DataFrame:
    uid_index = pd.Index(base["user_id"].unique(), name="user_id")
    acc = {
        "tot": pd.Series(0.0, index=uid_index),
        "comp": pd.Series(0.0, index=uid_index),
        "nsfw": pd.Series(0.0, index=uid_index),
        "fail": pd.Series(0.0, index=uid_index),
        "canc": pd.Series(0.0, index=uid_index),
        "d0": pd.Series(0.0, index=uid_index),
        "d13": pd.Series(0.0, index=uid_index),
    }
    n_chunks = 0
    for ch in pd.read_csv(
        gen_path,
        chunksize=CHUNK,
        usecols=["user_id", "created_at", "status"],
        low_memory=False,
    ):
        n_chunks += 1
        ch = ch.merge(base[["user_id", "sub_start"]], on="user_id", how="inner")
        ch = ch.dropna(subset=["sub_start"])
        cr = [isoparse(str(x)) if pd.notna(x) else pd.NaT for x in ch["created_at"]]
        ch["created"] = cr
        ch = ch.dropna(subset=["created"])
        subs = ch["sub_start"].tolist()
        crl = ch["created"].tolist()
        ch["days_from_sub"] = days_from_subscription(crl, subs)
        ch = ch.dropna(subset=["days_from_sub"])
        st = ch["status"].astype(str)
        ch["__comp"] = (st == "completed").astype(int)
        ch["__nsfw"] = (st == "nsfw").astype(int)
        ch["__fail"] = (st == "failed").astype(int)
        ch["__canc"] = (st == "canceled").astype(int)
        ch["__d0"] = ((ch["days_from_sub"] >= 0) & (ch["days_from_sub"] < 1)).astype(int)
        ch["__d13"] = ((ch["days_from_sub"] >= 13) & (ch["days_from_sub"] < 14)).astype(int)
        ch["__n"] = 1
        g = ch.groupby("user_id")[
            ["__n", "__comp", "__nsfw", "__fail", "__canc", "__d0", "__d13"]
        ].sum()
        acc["tot"] = acc["tot"].add(g["__n"], fill_value=0)
        acc["comp"] = acc["comp"].add(g["__comp"], fill_value=0)
        acc["nsfw"] = acc["nsfw"].add(g["__nsfw"], fill_value=0)
        acc["fail"] = acc["fail"].add(g["__fail"], fill_value=0)
        acc["canc"] = acc["canc"].add(g["__canc"], fill_value=0)
        acc["d0"] = acc["d0"].add(g["__d0"], fill_value=0)
        acc["d13"] = acc["d13"].add(g["__d13"], fill_value=0)
        if n_chunks % 15 == 0:
            print("chunks", gen_path.name, n_chunks, flush=True)

    out = base.copy()
    m = out["user_id"].map
    n_tot = m(acc["tot"]).fillna(0).to_numpy(dtype=np.float64)
    n_comp = m(acc["comp"]).fillna(0).to_numpy(dtype=np.float64)
    n_nsfw = m(acc["nsfw"]).fillna(0).to_numpy(dtype=np.float64)
    d0 = m(acc["d0"]).fillna(0).to_numpy()
    d13 = m(acc["d13"]).fillna(0).to_numpy()
    out["total_generations"] = n_tot
    ok = (n_tot > 0) & np.isfinite(n_tot)
    nsfw_rate = np.zeros_like(n_tot, dtype=np.float64)
    success_ratio = np.zeros_like(n_tot, dtype=np.float64)
    np.divide(n_nsfw, n_tot, out=nsfw_rate, where=ok)
    np.divide(n_comp, n_tot, out=success_ratio, where=ok)
    out["nsfw_rate"] = nsfw_rate
    out["success_ratio"] = success_ratio
    out["gen_delta_day1_minus_day14"] = d0 - d13
    out["log1p_total_gen"] = np.log1p(n_tot.astype(float))
    return out


def load_base_users_props(users_path: Path, props_path: Path) -> pd.DataFrame:
    users = read_csv_drop_index(users_path)
    props = read_csv_drop_index(props_path)
    base = users.merge(props, on="user_id", how="left", validate="one_to_one")
    base["sub_start"] = base["subscription_start_date"].map(parse_subscription_ts)
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="Add generation-based features from preprocessed CSVs.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Root containing train/ and test/ (same layout as preprocess_data output).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write CSVs (default: INPUT_DIR/cache).",
    )
    p.add_argument(
        "--train-out",
        type=Path,
        default=None,
        help="Train output CSV path (default: OUTPUT_DIR/train_users_with_generations.csv).",
    )
    p.add_argument(
        "--test-out",
        type=Path,
        default=None,
        help="Test output CSV path (default: OUTPUT_DIR/test_users_with_generations.csv).",
    )
    args = p.parse_args()

    root = args.input_dir.resolve()
    train_dir = root / "train"
    test_dir = root / "test"

    train_users = train_dir / "train_users.csv"
    train_props = train_dir / "train_users_properties.csv"
    train_gen = train_dir / "train_users_generations.csv"

    test_users = test_dir / "test_users.csv"
    test_props = test_dir / "test_users_properties.csv"
    test_gen = test_dir / "test_users_generations.csv"

    for path in (train_users, train_props, train_gen, test_users, test_props, test_gen):
        if not path.is_file():
            print(f"Missing file: {path}", file=sys.stderr)
            sys.exit(1)

    out_dir = (args.output_dir or (root / "cache")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.train_out or (out_dir / "train_users_with_generations.csv")
    test_out = args.test_out or (out_dir / "test_users_with_generations.csv")

    print("Loading train users + properties...")
    base_train = load_base_users_props(train_users, train_props)
    if "churn_status" in base_train.columns:
        base_train["churn_binary"] = base_train["churn_status"].isin(["vol_churn", "invol_churn"]).astype(int)

    print("Loading test users + properties...")
    base_test = load_base_users_props(test_users, test_props)

    print("Aggregating train generations (chunked)...")
    base_train = aggregate_generations(base_train, train_gen)
    print("Aggregating test generations (chunked)...")
    base_test = aggregate_generations(base_test, test_gen)

    print("train total_generations median:", float(np.median(base_train["total_generations"])))
    print("test total_generations median:", float(np.median(base_test["total_generations"])))

    base_train.to_csv(train_out, index=False)
    base_test.to_csv(test_out, index=False)
    print("Wrote:", train_out)
    print("Wrote:", test_out)


if __name__ == "__main__":
    main()
