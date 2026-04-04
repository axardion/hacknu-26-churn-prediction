#!/usr/bin/env python3
"""
Append column ``share_video_model_7_times_log1p_gen_total`` to train/test user CSVs.

Within-user share of ``video_model_7`` (count / gen_total, 0 if gen_total==0) times
``log1p(gen_total)``. Same CLI as other feature_* scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from add_features_generations_common import (
    aggregate_generation_type_wide,
    gen_cnt_series,
    parse_io_args,
    run_inplace_update,
    validate_inputs,
)

FEATURE_NAME = "share_video_model_7_times_log1p_gen_total"
VIDEO_MODEL_7 = "video_model_7"


def build_df(base: pd.DataFrame, gen_path: Path) -> pd.DataFrame:
    w = aggregate_generation_type_wide(base, gen_path)
    gt = w["gen_total"].to_numpy(dtype=np.float64)
    cnt_v7 = gen_cnt_series(w, VIDEO_MODEL_7).to_numpy(dtype=np.float64)
    share = np.divide(cnt_v7, gt, out=np.zeros_like(gt), where=gt > 0)
    feat = share * np.log1p(gt)
    out = w[["user_id"]].copy()
    out[FEATURE_NAME] = feat
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
