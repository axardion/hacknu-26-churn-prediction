#!/usr/bin/env python3
"""
Chunk-read generations CSVs and append generation features as new columns on the user tables.

By default **overwrites**:
  - data/train/train_users.csv
  - data/test/test_users.csv

Still reads properties + generations from the same train/ and test/ folders (for subscription
time and chunked generation rows). Same aggregation logic as churn_importance_by_time_bin.

New columns (if missing; existing names are replaced):
  total_generations, nsfw_rate, success_ratio, gen_delta_day1_minus_day14, log1p_total_gen
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

GEN_FEATURE_COLS = [
    "total_generations",
    "nsfw_rate",
    "success_ratio",
    "gen_delta_day1_minus_day14",
    "log1p_total_gen",
]


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


def merge_gen_into_users(users_df: pd.DataFrame, gen_aggregated: pd.DataFrame) -> pd.DataFrame:
    """Left-join generation columns onto the original user rows; drop old gen cols if present."""
    out = users_df.copy()
    for c in GEN_FEATURE_COLS:
        if c in out.columns:
            out = out.drop(columns=[c])
    part = gen_aggregated[["user_id"] + GEN_FEATURE_COLS].drop_duplicates(subset=["user_id"], keep="first")
    return out.merge(part, on="user_id", how="left")


def main() -> None:
    root = Path("data")
    p = argparse.ArgumentParser(
        description="Append generation feature columns onto train_users.csv and test_users.csv."
    )
    p.add_argument(
        "--train-users",
        type=Path,
        default=root / "train" / "train_users.csv",
        help="Train user table to update in place.",
    )
    p.add_argument(
        "--test-users",
        type=Path,
        default=root / "test" / "test_users.csv",
        help="Test user table to update in place.",
    )
    p.add_argument(
        "--train-props",
        type=Path,
        default=root / "train" / "train_users_properties.csv",
        help="Train properties (subscription_start_date for timing).",
    )
    p.add_argument(
        "--test-props",
        type=Path,
        default=root / "test" / "test_users_properties.csv",
        help="Test properties.",
    )
    p.add_argument(
        "--train-gen",
        type=Path,
        default=root / "train" / "train_users_generations.csv",
        help="Train generations log.",
    )
    p.add_argument(
        "--test-gen",
        type=Path,
        default=root / "test" / "test_users_generations.csv",
        help="Test generations log.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features and print stats but do not write CSVs.",
    )
    args = p.parse_args()

    train_users_path = args.train_users.resolve()
    test_users_path = args.test_users.resolve()
    train_props = args.train_props.resolve()
    test_props = args.test_props.resolve()
    train_gen = args.train_gen.resolve()
    test_gen = args.test_gen.resolve()

    for path in (train_users_path, test_users_path, train_props, test_props, train_gen, test_gen):
        if not path.is_file():
            print(f"Missing file: {path}", file=sys.stderr)
            sys.exit(1)

    print("Loading train users + properties (for aggregation)...")
    base_train = load_base_users_props(train_users_path, train_props)
    print("Loading test users + properties...")
    base_test = load_base_users_props(test_users_path, test_props)

    print("Aggregating train generations (chunked)...")
    base_train = aggregate_generations(base_train, train_gen)
    print("Aggregating test generations (chunked)...")
    base_test = aggregate_generations(base_test, test_gen)

    print("train total_generations median:", float(np.median(base_train["total_generations"])))
    print("test total_generations median:", float(np.median(base_test["total_generations"])))

    users_train = read_csv_drop_index(train_users_path)
    users_test = read_csv_drop_index(test_users_path)

    out_train = merge_gen_into_users(users_train, base_train)
    out_test = merge_gen_into_users(users_test, base_test)

    if args.dry_run:
        print("Dry run: not writing files.")
        print("Train columns:", list(out_train.columns))
        print("Test columns:", list(out_test.columns))
        return

    out_train.to_csv(train_users_path, index=False)
    out_test.to_csv(test_users_path, index=False)
    print("Updated:", train_users_path)
    print("Updated:", test_users_path)
    print("Appended columns:", GEN_FEATURE_COLS)


if __name__ == "__main__":
    main()
