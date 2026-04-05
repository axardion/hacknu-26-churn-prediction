#!/usr/bin/env python3
"""
Merge train CSVs from ``data/preprocessed/train`` and ``data/train`` into ``data/merged/train``.

For each table, rows are aligned on the natural key(s) and overlapping column names get
suffixes ``_prep`` (preprocessed) and ``_raw`` (raw ``data/train``). Columns that exist
only on one side are included without a suffix.

Examples::

    python scripts/merge_train_sources.py
    python scripts/merge_train_sources.py --only train_users.csv train_users_quizzes.csv
    python scripts/merge_train_sources.py --include-generations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PREPROCESSED = Path(__file__).resolve().parent.parent / "data" / "preprocessed" / "train"
DEFAULT_RAW = Path(__file__).resolve().parent.parent / "data" / "train"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "merged" / "train"

# (filename, merge keys)
TABLE_CONFIG: list[tuple[str, list[str]]] = [
    ("train_users.csv", ["user_id"]),
    ("train_users_properties.csv", ["user_id"]),
    ("train_users_quizzes.csv", ["user_id"]),
    ("train_users_purchases.csv", ["user_id", "transaction_id"]),
    ("train_users_transaction_attempts.csv", ["transaction_id"]),
    ("train_users_generations.csv", ["user_id", "generation_id"]),
]


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    c0 = df.columns[0]
    if str(c0).startswith("Unnamed") or str(c0).lower() == "skipped":
        df = df.drop(columns=c0)
    return df


def normalize_keys(df: pd.DataFrame, keys: list[str]) -> None:
    for k in keys:
        if k in df.columns:
            df[k] = df[k].astype(str)


def merge_pair(
    prep: pd.DataFrame,
    raw: pd.DataFrame,
    on_keys: list[str],
) -> pd.DataFrame:
    normalize_keys(prep, on_keys)
    normalize_keys(raw, on_keys)
    missing_prep = [k for k in on_keys if k not in prep.columns]
    missing_raw = [k for k in on_keys if k not in raw.columns]
    if missing_prep or missing_raw:
        raise ValueError(
            f"Missing merge keys: preprocessed {missing_prep!r}, raw {missing_raw!r}"
        )
    return pd.merge(
        prep,
        raw,
        on=on_keys,
        how="outer",
        suffixes=("_prep", "_raw"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=DEFAULT_PREPROCESSED,
        help=f"Default: {DEFAULT_PREPROCESSED}",
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help=f"Default: {DEFAULT_RAW}",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Default: {DEFAULT_OUT}",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        metavar="FILE",
        help="If set, only merge these basenames (e.g. train_users.csv).",
    )
    ap.add_argument(
        "--include-generations",
        action="store_true",
        help="Also merge train_users_generations.csv (very large; high memory use).",
    )
    args = ap.parse_args()

    prep_dir: Path = args.preprocessed_dir
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only) if args.only else None
    tables = [
        (fn, keys)
        for fn, keys in TABLE_CONFIG
        if (only is None or fn in only)
        and (args.include_generations or fn != "train_users_generations.csv")
    ]

    if only is not None:
        unknown = only - {fn for fn, _ in TABLE_CONFIG}
        if unknown:
            print(f"Unknown --only names: {sorted(unknown)}", file=sys.stderr)
            return 1

    for filename, keys in tables:
        p_prep = prep_dir / filename
        p_raw = raw_dir / filename
        if not p_prep.is_file():
            print(f"Skip (missing preprocessed): {p_prep}", file=sys.stderr)
            continue
        if not p_raw.is_file():
            print(f"Skip (missing raw): {p_raw}", file=sys.stderr)
            continue

        print(f"Merging {filename} on {keys} ...")
        df_prep = read_csv_clean(p_prep)
        df_raw = read_csv_clean(p_raw)
        merged = merge_pair(df_prep, df_raw, keys)
        out_path = out_dir / filename
        merged.to_csv(out_path, index=False)
        print(f"  -> {out_path}  ({len(merged)} rows, {len(merged.columns)} cols)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
