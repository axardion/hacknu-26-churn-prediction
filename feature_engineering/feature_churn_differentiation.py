#!/usr/bin/env python3
"""
Append churn-differentiation interaction columns to train/test ``*_users.csv``:

- ``inter_payment_risk_score``
- ``inter_mismatch``
- ``inter_country_fail_rate``
- ``inter_3ds_x_geohigh``
- ``inter_debit_x_geohigh``

Computed via ``churn_analysis.build_inter_features_without_generations`` (same logic as
vol/invol ``inter_*`` signals; no generations file required).

Defaults target ``data/preprocessed/train`` and ``data/preprocessed/test``.

Run::

     python feature_engineering/feature_churn_differentiation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_FE_DIR = Path(__file__).resolve().parent
if str(_FE_DIR) not in sys.path:
    sys.path.insert(0, str(_FE_DIR))

import pandas as pd

from add_features_generations_common import read_csv_drop_index, validate_inputs
from churn_analysis import build_inter_features_without_generations

FEATURE_NAMES = [
    "inter_payment_risk_score",
    "inter_mismatch",
    "inter_country_fail_rate",
    "inter_3ds_x_geohigh",
    "inter_debit_x_geohigh",
]

_REPO_ROOT = _FE_DIR.parent


def build_feat_df(
    users_path: Path,
    props_path: Path,
    purchases_path: Path,
    txn_path: Path,
    quizzes_path: Path,
) -> pd.DataFrame:
    full = build_inter_features_without_generations(
        users_path, props_path, purchases_path, txn_path, quizzes_path
    )
    missing = [c for c in FEATURE_NAMES if c not in full.columns]
    if missing:
        raise RuntimeError(f"Expected columns missing from interaction build: {missing}")
    out = full[["user_id"] + FEATURE_NAMES].copy()
    out["user_id"] = out["user_id"].astype(str)
    return out


def merge_features_into_users(
    users_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    names: list[str],
) -> pd.DataFrame:
    out = users_df.copy()
    for fn in names:
        if fn in out.columns:
            out = out.drop(columns=[fn])
    cols = ["user_id"] + [c for c in names if c in feat_df.columns]
    part = feat_df[cols].drop_duplicates(subset=["user_id"], keep="first")
    out["user_id"] = out["user_id"].astype(str)
    return out.merge(part, on="user_id", how="left")


def parse_args():
    pre = _REPO_ROOT / "data" / "preprocessed"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-users", type=Path, default=pre / "train" / "train_users.csv")
    p.add_argument("--test-users", type=Path, default=pre / "test" / "test_users.csv")
    p.add_argument("--train-props", type=Path, default=pre / "train" / "train_users_properties.csv")
    p.add_argument("--test-props", type=Path, default=pre / "test" / "test_users_properties.csv")
    p.add_argument(
        "--train-purchases",
        type=Path,
        default=pre / "train" / "train_users_purchases.csv",
    )
    p.add_argument(
        "--test-purchases",
        type=Path,
        default=pre / "test" / "test_users_purchases.csv",
    )
    p.add_argument(
        "--train-txn",
        type=Path,
        default=pre / "train" / "train_users_transaction_attempts.csv",
    )
    p.add_argument(
        "--test-txn",
        type=Path,
        default=pre / "test" / "test_users_transaction_attempts.csv",
    )
    p.add_argument(
        "--train-quizzes",
        type=Path,
        default=pre / "train" / "train_users_quizzes.csv",
    )
    p.add_argument(
        "--test-quizzes",
        type=Path,
        default=pre / "test" / "test_users_quizzes.csv",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_users = args.train_users.resolve()
    test_users = args.test_users.resolve()
    train_props = args.train_props.resolve()
    test_props = args.test_props.resolve()
    train_purchases = args.train_purchases.resolve()
    test_purchases = args.test_purchases.resolve()
    train_txn = args.train_txn.resolve()
    test_txn = args.test_txn.resolve()
    train_quizzes = args.train_quizzes.resolve()
    test_quizzes = args.test_quizzes.resolve()

    validate_inputs(
        {
            "train_users": train_users,
            "test_users": test_users,
            "train_props": train_props,
            "test_props": test_props,
            "train_purchases": train_purchases,
            "test_purchases": test_purchases,
            "train_txn": train_txn,
            "test_txn": test_txn,
            "train_quizzes": train_quizzes,
            "test_quizzes": test_quizzes,
        }
    )

    print("Feature columns:", FEATURE_NAMES)
    print("Building interaction features (train)...")
    df_tr = build_feat_df(
        train_users, train_props, train_purchases, train_txn, train_quizzes
    )
    print("Building interaction features (test)...")
    df_te = build_feat_df(
        test_users, test_props, test_purchases, test_txn, test_quizzes
    )

    u_train = read_csv_drop_index(train_users)
    u_test = read_csv_drop_index(test_users)

    out_train = merge_features_into_users(u_train, df_tr, FEATURE_NAMES)
    out_test = merge_features_into_users(u_test, df_te, FEATURE_NAMES)

    if args.dry_run:
        print("Dry run: not writing files.")
        print("Train columns:", list(out_train.columns))
        print("Test columns:", list(out_test.columns))
        print(out_train[FEATURE_NAMES].head(3))
        return

    out_train.to_csv(train_users, index=False)
    out_test.to_csv(test_users, index=False)
    print("Updated:", train_users)
    print("Updated:", test_users)
    print("Appended:", ", ".join(FEATURE_NAMES))


if __name__ == "__main__":
    raise SystemExit(main())
