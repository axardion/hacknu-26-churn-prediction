#!/usr/bin/env python3
"""
Append column ``has_any_generation`` to train/test user CSVs.

Binary 1 if the user has at least one generation row (gen_total > 0), else 0.
Complements mix/share features for inactive users. Same CLI as other feature_* scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from add_features_generations_common import (
    aggregate_generation_type_wide,
    parse_io_args,
    run_inplace_update,
    validate_inputs,
)

FEATURE_NAME = "has_any_generation"


def build_df(base: pd.DataFrame, gen_path: Path) -> pd.DataFrame:
    w = aggregate_generation_type_wide(base, gen_path)
    gt = w["gen_total"].to_numpy(dtype=np.float64)
    out = w[["user_id"]].copy()
    out[FEATURE_NAME] = (gt > 0).astype(np.int8)
    return out


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
