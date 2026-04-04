#!/usr/bin/env python3
"""
Append column ``gen_duration_median_video`` to data/train/train_users.csv and data/test/test_users.csv.

Median ``duration`` over rows with ``generation_type`` starting with ``video_`` (DuckDB streaming).
Requires ``pip install duckdb``. Same CLI flags as other feature_* scripts.
"""

from __future__ import annotations

import pandas as pd

from add_features_generations_common import (
    aggregate_duration_median_for_modality_duckdb,
    parse_io_args,
    run_inplace_update,
    validate_inputs,
)

FEATURE_NAME = "gen_duration_median_video"


def build_df(base: pd.DataFrame, gen_path):
    return aggregate_duration_median_for_modality_duckdb(base, gen_path, "video")


def main() -> None:
    args = parse_io_args()
    train_users = args.train_users.resolve()
    test_users = args.test_users.resolve()
    train_props = args.train_props.resolve()
    test_props = args.test_props.resolve()
    train_gen = args.train_gen.resolve()
    test_gen = args.test_gen.resolve()
    validate_inputs(
        {
            "train_users": train_users,
            "test_users": test_users,
            "train_props": train_props,
            "test_props": test_props,
            "train_gen": train_gen,
            "test_gen": test_gen,
        }
    )
    run_inplace_update(
        FEATURE_NAME,
        build_df,
        dry_run=args.dry_run,
        train_users=train_users,
        test_users=test_users,
        train_props=train_props,
        test_props=test_props,
        train_gen=train_gen,
        test_gen=test_gen,
    )


if __name__ == "__main__":
    main()
