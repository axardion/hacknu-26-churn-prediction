#!/usr/bin/env python3
"""
Merge raw (`data/train`, `data/test`) with preprocessed (`data/preprocessed/train|test`)
CSV tables into `data/merged/train` and `data/merged/test`.

Each output file has the union of rows (outer join on keys) and all columns from both
sources; overlapping non-key column names get suffixes `_raw` and `_prep`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _drop_unnamed_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in list(out.columns):
        if c.startswith("Unnamed"):
            out = out.drop(columns=[c])
    return out


def merge_keys_for(filename: str) -> list[str]:
    """Primary key(s) for joining raw vs preprocessed tables."""
    name = filename.lower()
    if name.endswith("_users.csv"):
        return ["user_id"]
    if name.endswith("_users_properties.csv"):
        return ["user_id"]
    if name.endswith("_users_quizzes.csv"):
        return ["user_id"]
    if name.endswith("_users_purchases.csv"):
        return ["user_id", "transaction_id"]
    if name.endswith("_users_transaction_attempts.csv"):
        return ["transaction_id"]
    if name.endswith("_users_generations.csv"):
        return ["user_id", "generation_id"]
    raise ValueError(f"Unknown table type for merge keys: {filename}")


def merge_pair(raw_path: Path, prep_path: Path) -> pd.DataFrame:
    raw = _drop_unnamed_index(pd.read_csv(raw_path, low_memory=False))
    prep = _drop_unnamed_index(pd.read_csv(prep_path, low_memory=False))
    keys = merge_keys_for(raw_path.name)

    missing_l = [k for k in keys if k not in raw.columns]
    missing_r = [k for k in keys if k not in prep.columns]
    if missing_l or missing_r:
        raise ValueError(
            f"{raw_path.name}: missing keys raw={missing_l} prep={missing_r}"
        )

    merged = raw.merge(
        prep,
        on=keys,
        how="outer",
        suffixes=("_raw", "_prep"),
    )
    return merged


def process_split(data_root: Path, split: str) -> None:
    raw_dir = data_root / split
    prep_dir = data_root / "preprocessed" / split
    out_dir = data_root / "merged" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        print(f"Skip {split}: missing {raw_dir}", file=sys.stderr)
        return
    if not prep_dir.is_dir():
        print(f"Skip {split}: missing {prep_dir}", file=sys.stderr)
        return

    csvs = sorted(p for p in raw_dir.glob("*.csv") if p.is_file())
    n_ok = 0
    for raw_path in csvs:
        prep_path = prep_dir / raw_path.name
        if not prep_path.is_file():
            print(f"  [skip] no preprocessed file for {raw_path.name}", file=sys.stderr)
            continue
        try:
            merged = merge_pair(raw_path, prep_path)
        except ValueError as e:
            print(f"  [error] {raw_path.name}: {e}", file=sys.stderr)
            continue
        out_path = out_dir / raw_path.name
        merged.to_csv(out_path, index=False)
        print(f"  {raw_path.name}: {len(merged)} rows -> {out_path}")
        n_ok += 1
    print(f"Done {split}: wrote {n_ok} file(s) under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory containing train/, test/, preprocessed/ (default: ./data next to script)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Which split to merge (default: both)",
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if args.split in ("train", "both"):
        print("Merging TRAIN …")
        process_split(data_root, "train")
    if args.split in ("test", "both"):
        print("Merging TEST …")
        process_split(data_root, "test")


if __name__ == "__main__":
    main()
