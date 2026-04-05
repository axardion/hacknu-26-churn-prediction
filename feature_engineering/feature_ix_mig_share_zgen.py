#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_Z = "ix_zshare_x_zgen_delta"
MIG_ANY = "mig_any_b5_a5"
FEATURE_NAMES = [FEATURE_Z, MIG_ANY]


def _zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = s.median(skipna=True)
    s = s.fillna(m)
    sd = float(s.std(skipna=True))
    if not math.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index, dtype=np.float64)
    return (s - float(s.mean())) / sd


def load_mig_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["user_id"] = df["user_id"].astype(str)
    out = df[["user_id", MIG_ANY]].copy()
    print(f"    loaded {MIG_ANY} for n={len(out)} from {path.name}")
    return out


def merge_features_into_users(
    users_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    names: list[str],
) -> pd.DataFrame:
    out = users_df.copy()
    for legacy in (
        "ix_mig_any_b5_a5",
        "ix_mig_any_b5_a5_x_share",
        "ix_mig_trail15d_x_share",
    ):
        if legacy in out.columns:
            out = out.drop(columns=[legacy])
    for fn in names:
        if fn in out.columns:
            out = out.drop(columns=[fn])
    cols = ["user_id"] + [c for c in names if c in feat_df.columns]
    part = feat_df[cols].drop_duplicates(subset=["user_id"], keep="first")
    out["user_id"] = out["user_id"].astype(str)
    return out.merge(part, on="user_id", how="left")


def parse_args():
    alt = Path(__file__).resolve().parent.parent / "alt_data"
    p = argparse.ArgumentParser(description="Append mig_any_b5_a5 from pre-computed CSVs.")
    p.add_argument("--train-anchor", type=Path, default=alt / "train_user_market_anchor_features.csv")
    p.add_argument("--test-anchor", type=Path, default=alt / "test_user_market_anchor_features.csv")
    p.add_argument("--train-users", type=Path, default=None)
    p.add_argument("--test-users", type=Path, default=None)
    p.add_argument("--market", type=Path, default=alt / "market_features.csv")
    p.add_argument("--output-dir", type=Path, default=alt)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {MIG_ANY} from pre-computed anchor features...")
    mig_tr = load_mig_from_csv(args.train_anchor)
    mig_te = load_mig_from_csv(args.test_anchor)

    if args.train_users and args.test_users:
        u_train = pd.read_csv(args.train_users)
        u_test = pd.read_csv(args.test_users)
        available = [MIG_ANY]
        out_train = merge_features_into_users(u_train, mig_tr, available)
        out_test = merge_features_into_users(u_test, mig_te, available)

        if args.dry_run:
            print("Dry run: not writing files.")
            print("Train columns:", list(out_train.columns))
            print("Test columns:", list(out_test.columns))
            print(out_train[[MIG_ANY]].head(3))
            return

        out_train.to_csv(args.train_users, index=False)
        out_test.to_csv(args.test_users, index=False)
        print("Updated:", args.train_users)
        print("Updated:", args.test_users)
    else:
        out_dir = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        train_out = out_dir / "train_mig_features.csv"
        test_out = out_dir / "test_mig_features.csv"

        if args.dry_run:
            print("Dry run: not writing files.")
            print("Train sample:")
            print(mig_tr.head(3))
            print("Test sample:")
            print(mig_te.head(3))
            return

        mig_tr.to_csv(train_out, index=False)
        mig_te.to_csv(test_out, index=False)
        print("Wrote:", train_out)
        print("Wrote:", test_out)

    print(f"Note: {FEATURE_Z} requires generation data not in these CSVs — skipped.")
    print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())