"""
Shared helpers for feature_* generation scripts (chunked reads, merge into train/test users).
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Literal
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
    "share_video_model_7_times_log1p_gen_total",
    "video_gen_share",
    "has_any_generation",
    "gen_duration_mean_image",
    "gen_duration_mean_video",
    "gen_duration_median_image",
    "gen_duration_median_video",
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


def load_base_users_props(users_path: Path, props_path: Path) -> pd.DataFrame:
    users = read_csv_drop_index(users_path)
    props = read_csv_drop_index(props_path)
    base = users.merge(props, on="user_id", how="left", validate="one_to_one")
    base["sub_start"] = base["subscription_start_date"].map(parse_subscription_ts)
    return base


def aggregate_generations(base: pd.DataFrame, gen_path: Path) -> pd.DataFrame:
    """Full chunked aggregation (same as original add_features_from_generations)."""
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


def aggregate_generation_type_wide(base: pd.DataFrame, gen_path: Path) -> pd.DataFrame:
    """Chunked user × generation_type counts merged onto `base` users.

    Adds ``gen_total`` and ``gen_cnt__<generation_type>`` columns (zeros when missing).
    Only rows with ``user_id`` in ``base`` are read from the generations file.
    """
    uid_set = set(base["user_id"].astype(str))
    parts: list[pd.DataFrame] = []
    n_chunks = 0
    for ch in pd.read_csv(
        gen_path,
        chunksize=CHUNK,
        usecols=["user_id", "generation_type"],
        low_memory=False,
        dtype={"user_id": str, "generation_type": str},
    ):
        n_chunks += 1
        ch = ch[ch["user_id"].isin(uid_set)]
        if ch.empty:
            continue
        ch["generation_type"] = ch["generation_type"].fillna("__NA__")
        g = ch.groupby(["user_id", "generation_type"], observed=False).size().reset_index(name="n")
        parts.append(g)
        if n_chunks % 20 == 0:
            print("  gen_type chunks", gen_path.name, n_chunks, flush=True)

    out = base[["user_id"]].copy()
    if not parts:
        out["gen_total"] = 0.0
        return out

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.groupby(["user_id", "generation_type"], observed=False)["n"].sum().reset_index()
    wide = long_df.pivot(index="user_id", columns="generation_type", values="n").fillna(0.0)
    wide = wide.rename(columns={c: f"gen_cnt__{c}" for c in wide.columns})
    wide["gen_total"] = wide.sum(axis=1)
    wide = wide.reset_index()
    merged = out.merge(wide, on="user_id", how="left")
    for c in merged.columns:
        if c.startswith("gen_cnt__") or c == "gen_total":
            merged[c] = merged[c].fillna(0.0)
    return merged


def gen_cnt_series(df: pd.DataFrame, generation_type: str) -> pd.Series:
    """Column ``gen_cnt__{generation_type}`` if present, else zeros."""
    c = f"gen_cnt__{generation_type}"
    if c in df.columns:
        return df[c].astype(np.float64)
    return pd.Series(0.0, index=df.index, dtype=np.float64)


def video_gen_count_sum(df: pd.DataFrame) -> pd.Series:
    """Sum of ``gen_cnt__*`` whose type name starts with ``video_``."""
    cols = [
        c
        for c in df.columns
        if c.startswith("gen_cnt__") and c.replace("gen_cnt__", "", 1).startswith("video_")
    ]
    if not cols:
        return pd.Series(0.0, index=df.index, dtype=np.float64)
    return df[cols].sum(axis=1).astype(np.float64)


def _duration_seconds_with_timestamp_fallback(ch: pd.DataFrame) -> pd.DataFrame:
    """Set ``dur`` from ``duration`` when numeric; else from ``completed_at - created_at`` (seconds).

    Image rows often omit ``duration`` in the export; wall-clock gap is a usable proxy.
    Ignores negative or absurd gaps (>7 days).

    Uses ``dateutil.isoparse`` for timestamps: exports may use years outside pandas' datetime
    range (e.g. year 1067), where ``pd.to_datetime`` becomes NaT and would drop all rows.
    """
    ch = ch.copy()
    ch["dur"] = pd.to_numeric(ch["duration"], errors="coerce")
    if "created_at" not in ch.columns or "completed_at" not in ch.columns:
        return ch

    need_fb = ch["dur"].isna().to_numpy()
    if not need_fb.any():
        return ch

    cr = ch["created_at"].to_numpy()
    co = ch["completed_at"].to_numpy()
    n = len(ch)
    deltas = np.full(n, np.nan, dtype=np.float64)
    max_s = 7 * 86400.0
    for i in range(n):
        if not need_fb[i]:
            continue
        try:
            if pd.isna(cr[i]) or pd.isna(co[i]):
                continue
            c = isoparse(str(cr[i]))
            e = isoparse(str(co[i]))
            d = (e - c).total_seconds()
            if 0.0 <= d <= max_s:
                deltas[i] = d
        except (ValueError, TypeError, OverflowError):
            continue

    fill = need_fb & np.isfinite(deltas)
    ch.loc[fill, "dur"] = deltas[fill]
    return ch


def aggregate_duration_mean_for_modality(
    base: pd.DataFrame, gen_path: Path, modality: Literal["image", "video"]
) -> pd.DataFrame:
    """Mean duration (seconds) per user for ``image_`` or ``video_`` ``generation_type`` rows.

    Uses numeric ``duration`` when present; otherwise ``completed_at - created_at`` when both
    timestamps exist (common for image generations with empty ``duration``). Users with no
    usable rows for that modality get NaN.
    """
    col = "gen_duration_mean_image" if modality == "image" else "gen_duration_mean_video"
    prefix = "image_" if modality == "image" else "video_"

    uid_set = set(base["user_id"].astype(str))
    uid_index = pd.Index(base["user_id"].unique(), name="user_id")
    sum_d = pd.Series(0.0, index=uid_index, dtype=np.float64)
    cnt_d = pd.Series(0.0, index=uid_index, dtype=np.float64)

    n_chunks = 0
    for ch in pd.read_csv(
        gen_path,
        chunksize=CHUNK,
        usecols=["user_id", "generation_type", "duration", "created_at", "completed_at"],
        low_memory=False,
        dtype={"user_id": str, "generation_type": str},
    ):
        n_chunks += 1
        ch = ch[ch["user_id"].isin(uid_set)]
        if ch.empty:
            continue
        ch = _duration_seconds_with_timestamp_fallback(ch)
        ch = ch.dropna(subset=["dur"])
        if ch.empty:
            continue
        gt = ch["generation_type"].fillna("").astype(str)
        sub = ch[gt.str.startswith(prefix)]
        if not sub.empty:
            g = sub.groupby("user_id")["dur"].agg(["sum", "count"])
            sum_d = sum_d.add(g["sum"], fill_value=0)
            cnt_d = cnt_d.add(g["count"], fill_value=0)
        if n_chunks % 20 == 0:
            print("  duration mean chunks", gen_path.name, n_chunks, flush=True)

    out = base[["user_id"]].copy()
    sm = sum_d.loc[out["user_id"]].to_numpy(dtype=np.float64)
    cn = cnt_d.loc[out["user_id"]].to_numpy(dtype=np.float64)
    # Avoid sm/cn where cn==0 (np.where still evaluates both branches and triggers divide warnings).
    mean_val = np.full(sm.shape, np.nan, dtype=np.float64)
    np.divide(sm, cn, out=mean_val, where=cn > 0)
    out[col] = mean_val
    return out


def aggregate_duration_median_for_modality_duckdb(
    base: pd.DataFrame, gen_path: Path, modality: Literal["image", "video"]
) -> pd.DataFrame:
    """Median duration (seconds) per user for one modality (streaming CSV via DuckDB).

    Same definition as ``aggregate_duration_mean_for_modality``: numeric ``duration`` if present,
    else ``completed_at - created_at`` in seconds when both parse as timestamps (capped 0–7 days).
    """
    try:
        import duckdb
    except ImportError as e:
        raise RuntimeError(
            "Median aggregation needs the duckdb package: pip install duckdb"
        ) from e

    col = "gen_duration_median_image" if modality == "image" else "gen_duration_median_video"
    prefix = "image_" if modality == "image" else "video_"

    gen_path = gen_path.resolve()
    if not gen_path.is_file():
        raise FileNotFoundError(gen_path)

    con = duckdb.connect(database=":memory:")
    con.register("base_users", base[["user_id"]].copy())
    gpath = str(gen_path).replace("\\", "/")
    q = f"""
    SELECT g.user_id,
      median(
        CASE
          WHEN COALESCE(
            try_cast(g.duration AS DOUBLE),
            CASE
              WHEN try_cast(g.created_at AS TIMESTAMP) IS NOT NULL
                AND try_cast(g.completed_at AS TIMESTAMP) IS NOT NULL
              THEN date_diff(
                'microsecond',
                try_cast(g.created_at AS TIMESTAMP),
                try_cast(g.completed_at AS TIMESTAMP)
              ) / 1000000.0
              ELSE NULL
            END
          ) BETWEEN 0 AND 604800
          THEN COALESCE(
            try_cast(g.duration AS DOUBLE),
            CASE
              WHEN try_cast(g.created_at AS TIMESTAMP) IS NOT NULL
                AND try_cast(g.completed_at AS TIMESTAMP) IS NOT NULL
              THEN date_diff(
                'microsecond',
                try_cast(g.created_at AS TIMESTAMP),
                try_cast(g.completed_at AS TIMESTAMP)
              ) / 1000000.0
              ELSE NULL
            END
          )
          ELSE NULL
        END
      ) AS {col}
    FROM read_csv_auto(?) g
    INNER JOIN base_users b ON g.user_id = b.user_id
    WHERE starts_with(CAST(g.generation_type AS VARCHAR), '{prefix}')
    GROUP BY g.user_id
    """
    sub = con.execute(q, [gpath]).df()
    out = base[["user_id"]].merge(sub, on="user_id", how="left")
    return out


def merge_feature_into_users(users_df: pd.DataFrame, feature_name: str, gen_df: pd.DataFrame) -> pd.DataFrame:
    """Keep every column already in `users_df` (your CSV); add or refresh one feature column.

    If `feature_name` already exists, that column is replaced with newly computed values;
    all other columns are unchanged.
    """
    out = users_df.copy()
    if feature_name in out.columns:
        out = out.drop(columns=[feature_name])
    part = gen_df[["user_id", feature_name]].drop_duplicates(subset=["user_id"], keep="first")
    return out.merge(part, on="user_id", how="left")


def default_paths():
    root = Path("data")
    return {
        "train_users": root / "train" / "train_users.csv",
        "test_users": root / "test" / "test_users.csv",
        "train_props": root / "train" / "train_users_properties.csv",
        "test_props": root / "test" / "test_users_properties.csv",
        "train_gen": root / "train" / "train_users_generations.csv",
        "test_gen": root / "test" / "test_users_generations.csv",
    }


def parse_io_args():
    root = Path("data")
    p = argparse.ArgumentParser()
    p.add_argument("--train-users", type=Path, default=root / "train" / "train_users.csv")
    p.add_argument("--test-users", type=Path, default=root / "test" / "test_users.csv")
    p.add_argument("--train-props", type=Path, default=root / "train" / "train_users_properties.csv")
    p.add_argument("--test-props", type=Path, default=root / "test" / "test_users_properties.csv")
    p.add_argument("--train-gen", type=Path, default=root / "train" / "train_users_generations.csv")
    p.add_argument("--test-gen", type=Path, default=root / "test" / "test_users_generations.csv")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def validate_inputs(paths: dict) -> None:
    for k, path in paths.items():
        if not path.is_file():
            print(f"Missing file: {path}", file=sys.stderr)
            sys.exit(1)


def run_inplace_update(
    feature_name: str,
    build_df: callable,
    *,
    dry_run: bool,
    train_users: Path,
    test_users: Path,
    train_props: Path,
    test_props: Path,
    train_gen: Path,
    test_gen: Path,
) -> None:
    """Load bases, build single-feature dataframe (user_id + column), merge, write."""
    print(f"Feature: {feature_name}")
    print("Loading train users + properties...")
    base_train = load_base_users_props(train_users, train_props)
    print("Loading test users + properties...")
    base_test = load_base_users_props(test_users, test_props)

    print("Aggregating train (chunked)...")
    df_tr = build_df(base_train, train_gen)
    print("Aggregating test (chunked)...")
    df_te = build_df(base_test, test_gen)

    users_train = read_csv_drop_index(train_users)
    users_test = read_csv_drop_index(test_users)

    out_train = merge_feature_into_users(users_train, feature_name, df_tr)
    out_test = merge_feature_into_users(users_test, feature_name, df_te)

    if dry_run:
        print("Dry run: not writing files.")
        print("Train columns:", list(out_train.columns))
        print("Test columns:", list(out_test.columns))
        return

    out_train.to_csv(train_users, index=False)
    out_test.to_csv(test_users, index=False)
    print("Updated:", train_users)
    print("Updated:", test_users)
    print("Appended column:", feature_name)


def build_single_column_df(base: pd.DataFrame, gen_path: Path, column: str) -> pd.DataFrame:
    """Internal helper: after aggregation, keep only user_id + `column` for the join.

    This does not remove columns from train_users.csv; it only narrows the temporary
    dataframe used to supply one new column when merging.
    """
    full = aggregate_generations(base, gen_path)
    return full[["user_id", column]].copy()


def main_for_feature(feature_name: str) -> None:
    """CLI entry for feature_<name>.py scripts."""
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

    def build_df(base: pd.DataFrame, gen_path: Path) -> pd.DataFrame:
        return build_single_column_df(base, gen_path, feature_name)

    run_inplace_update(
        feature_name,
        build_df,
        dry_run=args.dry_run,
        train_users=train_users,
        test_users=test_users,
        train_props=train_props,
        test_props=test_props,
        train_gen=train_gen,
        test_gen=test_gen,
    )
