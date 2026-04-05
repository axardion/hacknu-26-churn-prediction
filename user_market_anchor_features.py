#!/usr/bin/env python3
"""
User-level market features anchored on subscription_start_date.

Inputs:
  - data/social_mentions.csv (YouTube rows from competitors_features.YouTubeCollector)
  - data/google_trends.csv
  - data/github_stars.csv, data/huggingface_downloads.csv (optional OSS context)
  - data/train/train_users_properties.csv (and/or test)

How is_migration_signal produced?
  In competitors_features.YouTubeCollector, each video title is lowercased and:
    is_migration_signal = any(kw in title for kw in ["switch", "leaving", "alternative", "moved to"])
  So it is a cheap keyword flag — NOT sentiment, NOT "user churned". It has:
    - High precision for explicit "switching/leaving/alternative" language
    - Low recall: many comparison videos ("Higgsfield vs Runway") are False because
      they do not contain those exact words (see your CSV: most rows are False).
  Trust it as a noisy proxy for "migration narrative" videos, not ground truth.

GitHub / Hugging Face in this repo:
  Collectors spread repo totals across days (synthetic daily increments). They measure
  OSS ecosystem momentum (substitute tools), not per-user behavior. Use as GLOBAL
  context at the subscription date: rolling sums of stars_daily / downloads_daily
  ending on anchor day — weakly related to churn unless you believe OSS hype correlates
  with your cohort timing.

Decoding subscription_start_date (same idea as train_users_purchases + DateOffset):
  Real calendar time ≈ encoded_time + relativedelta(years=y, months=m, days=d)
  Defaults match the revenue script: +958 years, +5 days (SHIFT_YEARS=958, SHIFT_DAYS=5).
  Default path (months=0): vectorized year bump + pandas to_datetime — seconds on ~90k rows.
  Slow path (months≠0): dateutil + relativedelta per row. With months=0, `subscription_start_ts_raw`
  in the CSV output is the original encoded string (not a parsed Timestamp) to avoid 90k× isoparse.
  If the decoded anchor still falls outside ANCHOR_GRID_START..END_DATE, overlap features are NaN
  and sub_anchor_in_market_window = 0. ANCHOR_GRID_START is one day before churn START_DATE so
  2025-08-31 subscriptions align with the same external CSVs (Trends/social start 2025-09-01; Aug 31
  gets bfill/forward-filled where needed).
"""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta

from churn_features import END_DATE, START_DATE

# Include 2025-08-31 for subscription anchors (official churn window in churn_features is 2025-09-01..)
ANCHOR_GRID_START = pd.Timestamp(START_DATE) - pd.Timedelta(days=1)

DATA = Path("data")

# Same defaults as revenue dynamics script (train_users_purchases shift)
DEFAULT_SUB_SHIFT_DAYS = 5
DEFAULT_SUB_SHIFT_MONTHS = 0
DEFAULT_SUB_SHIFT_YEARS = 958


def _parse_subscription_cell(raw) -> pd.Timestamp:
    """Parse one subscription string to UTC. Uses dateutil so year 1067 works (pandas ns range cannot)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return pd.NaT
    s = str(raw).strip()
    if not s:
        return pd.NaT
    try:
        dt = isoparse(s)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return pd.Timestamp(dt)


def parse_subscription_series(series: pd.Series) -> pd.Series:
    return series.map(_parse_subscription_cell)


def decode_subscription_timestamps(
    ts: pd.Series,
    *,
    days: int = DEFAULT_SUB_SHIFT_DAYS,
    months: int = DEFAULT_SUB_SHIFT_MONTHS,
    years: int = DEFAULT_SUB_SHIFT_YEARS,
) -> pd.Series:
    """Same shift as purchases CSV: pd.DateOffset / relativedelta on decoded wall time."""

    def one(t: pd.Timestamp) -> pd.Timestamp:
        if pd.isna(t):
            return t
        py = t.to_pydatetime()
        shifted = py + relativedelta(years=years, months=months, days=days)
        st = pd.Timestamp(shifted)
        if st.tzinfo is not None:
            st = st.tz_convert("UTC")
        else:
            st = st.tz_localize("UTC")
        return st

    return ts.map(one)


def decode_subscription_vectorized(
    series: pd.Series,
    *,
    days: int = DEFAULT_SUB_SHIFT_DAYS,
    months: int = DEFAULT_SUB_SHIFT_MONTHS,
    years: int = DEFAULT_SUB_SHIFT_YEARS,
) -> pd.Series:
    """
    Fast path: bump ISO year in the string, then pandas to_datetime + Timedelta.
    Use when months==0 (your purchases script). ~100x faster than per-row relativedelta.
    """
    if months != 0:
        raise ValueError("Vectorized decode requires months=0; use slow path.")
    ser = series.astype(str)
    parts = ser.str.split("-", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.Series(pd.NaT, index=series.index)
    y = pd.to_numeric(parts[0], errors="coerce") + int(years)
    s2 = y.astype("Int64").astype(str) + "-" + parts[1]
    out = pd.to_datetime(s2, utc=True, errors="coerce")
    return out + pd.to_timedelta(int(days), unit="d")

# (days_before_subscription, days_after_subscription) inclusive on both ends
DEFAULT_REL_WINDOWS: list[tuple[int, int]] = [
    (3, 3),
    (5, 5),
    (3, 5),
    (6, 4),
    (4, 6),
    (7, 7),
    (0, 7),
    (7, 0),
]

# Trailing windows ending on anchor date (inclusive): last N days including anchor
ROLL_DAYS: list[int] = [3, 5, 7, 10, 15, 20, 30]


def _read_social(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def daily_migration_series(social: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-day counts from YouTube rows."""
    m = social[social["is_migration_signal"] == True].copy()  # noqa: E712
    mig = m.groupby("date").agg(
        mig_rows=("text", "count"),
        mig_engagement=("engagement", "sum"),
    )
    all_d = social.groupby("date").agg(
        all_rows=("text", "count"),
        all_engagement=("engagement", "sum"),
    )
    daily = mig.join(all_d, how="outer").fillna(0)
    daily["mig_share"] = daily["mig_rows"] / (daily["all_rows"] + 1e-9)
    return daily.sort_index()


def daily_trends(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, parse_dates=["date"])
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    t = t.set_index("date").sort_index()
    comp_cols = [c for c in t.columns if c != "higgsfield"]
    if comp_cols:
        t["comp_max"] = t[comp_cols].max(axis=1)
        t["comp_mean"] = t[comp_cols].mean(axis=1)
        t["hf_vs_comp_max"] = t["higgsfield"] / (t["comp_max"] + 1)
        tot = t[comp_cols].sum(axis=1) + t["higgsfield"]
        t["hf_share_voice"] = t["higgsfield"] / (tot + 1)
    return t


def daily_oss(path: Path, value_col: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.groupby("date")[value_col].sum().sort_index()


def _prefix_sum(arr: np.ndarray) -> np.ndarray:
    """csp[k] = sum(arr[0:k]); length len(arr)+1, csp[0]=0."""
    return np.concatenate((np.array([0.0], dtype=float), np.cumsum(arr.astype(float))))


def _sum_by_date_range(
    csp: np.ndarray,
    dates_np: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Inclusive date range [lo, hi] on daily grid dates_np."""
    ilo = np.searchsorted(dates_np, lo, side="left")
    ihi = np.searchsorted(dates_np, hi, side="right") - 1
    out = np.full(len(lo), np.nan, dtype=float)
    m = valid & (ilo <= ihi)
    out[m] = csp[ihi[m] + 1] - csp[ilo[m]]
    return out


def build_user_table(
    properties_path: Path,
    social_path: Path,
    trends_path: Path,
    github_path: Path | None,
    hf_path: Path | None,
    sub_shift_days: int = DEFAULT_SUB_SHIFT_DAYS,
    sub_shift_months: int = DEFAULT_SUB_SHIFT_MONTHS,
    sub_shift_years: int = DEFAULT_SUB_SHIFT_YEARS,
    decode_subscription: bool = True,
) -> pd.DataFrame:
    props = pd.read_csv(properties_path, low_memory=False)
    if props.columns[0].lower() in ("skipped", "unnamed: 0"):
        props = props.drop(columns=props.columns[0])
    if decode_subscription:
        if sub_shift_months == 0:
            props["subscription_start_ts"] = decode_subscription_vectorized(
                props["subscription_start_date"],
                days=sub_shift_days,
                months=sub_shift_months,
                years=sub_shift_years,
            )
            # Keep encoded values as strings (no 90k× isoparse — that was the 10+ min bottleneck)
            props["subscription_start_ts_raw"] = props["subscription_start_date"].astype(str)
        else:
            props["subscription_start_ts_raw"] = parse_subscription_series(
                props["subscription_start_date"]
            )
            props["subscription_start_ts"] = decode_subscription_timestamps(
                props["subscription_start_ts_raw"],
                days=sub_shift_days,
                months=sub_shift_months,
                years=sub_shift_years,
            )
    else:
        props["subscription_start_ts_raw"] = parse_subscription_series(props["subscription_start_date"])
        props["subscription_start_ts"] = props["subscription_start_ts_raw"]
    props["anchor_date"] = props["subscription_start_ts"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)

    social = _read_social(social_path)
    mig_daily = daily_migration_series(social)
    mig_count = mig_daily["mig_rows"].reindex(pd.date_range(ANCHOR_GRID_START, END_DATE, freq="D")).fillna(0)
    mig_eng = mig_daily["mig_engagement"].reindex(pd.date_range(ANCHOR_GRID_START, END_DATE, freq="D")).fillna(0)

    trends = daily_trends(trends_path)
    idx = pd.date_range(ANCHOR_GRID_START, END_DATE, freq="D")
    gt = trends.reindex(idx).ffill().bfill()

    gh_daily = daily_oss(github_path, "stars_daily") if github_path and github_path.exists() else pd.Series(dtype=float)
    hf_daily = daily_oss(hf_path, "downloads_daily") if hf_path and hf_path.exists() else pd.Series(dtype=float)

    gh_full = gh_daily.reindex(idx).fillna(0) if len(gh_daily) else pd.Series(0.0, index=idx)
    hf_full = hf_daily.reindex(idx).fillna(0) if len(hf_daily) else pd.Series(0.0, index=idx)

    out = props[
        ["user_id", "anchor_date", "subscription_start_ts_raw", "subscription_start_ts"]
    ].rename(columns={"subscription_start_ts": "subscription_start_ts_decoded"}).copy()
    out["sub_anchor_in_market_window"] = (
        out["anchor_date"].notna()
        & (out["anchor_date"] >= pd.Timestamp(ANCHOR_GRID_START))
        & (out["anchor_date"] <= pd.Timestamp(END_DATE))
    ).astype(int)

    dates_np = idx.values.astype("datetime64[ns]")
    t0 = dates_np[0]
    t1 = dates_np[-1]
    ad = out["anchor_date"].values.astype("datetime64[ns]")
    has_anch = out["anchor_date"].notna().values
    valid = has_anch & (out["anchor_date"].values >= pd.Timestamp(ANCHOR_GRID_START)) & (
        out["anchor_date"].values <= pd.Timestamp(END_DATE)
    )
    # Avoid NaT in timedelta / searchsorted (use dummy t0 where missing)
    ad_f = np.where(has_anch, ad, t0)

    v_mig = mig_count.reindex(idx).fillna(0).values.astype(float)
    v_eng = mig_eng.reindex(idx).fillna(0).values.astype(float)
    v_gh = gh_full.reindex(idx).fillna(0).values.astype(float)
    v_hf = hf_full.reindex(idx).fillna(0).values.astype(float)
    csp_mig = _prefix_sum(v_mig)
    csp_eng = _prefix_sum(v_eng)
    csp_gh = _prefix_sum(v_gh)
    csp_hf = _prefix_sum(v_hf)

    b_td = np.timedelta64(1, "D")

    for b, a in DEFAULT_REL_WINDOWS:
        lo = ad_f - np.int64(b) * b_td
        hi = ad_f + np.int64(a) * b_td
        lo = np.maximum(lo, t0)
        hi = np.minimum(hi, t1)
        vals_c = _sum_by_date_range(csp_mig, dates_np, lo, hi, valid)
        out[f"mig_count_b{b}_a{a}"] = vals_c
        out[f"mig_any_b{b}_a{a}"] = np.where(np.isnan(vals_c), np.nan, (vals_c > 0).astype(float))

    for n in ROLL_DAYS:
        lo = ad_f - np.int64(n - 1) * b_td
        hi = ad_f
        lo = np.maximum(lo, t0)
        hi = np.minimum(hi, t1)
        out[f"mig_sum_trailing_{n}d_at_sub"] = _sum_by_date_range(csp_mig, dates_np, lo, hi, valid)
        out[f"mig_eng_trailing_{n}d_at_sub"] = _sum_by_date_range(csp_eng, dates_np, lo, hi, valid)

    for n in ROLL_DAYS:
        cur_lo = np.maximum(ad_f - np.int64(n - 1) * b_td, t0)
        cur_hi = np.minimum(ad_f, t1)
        cur = _sum_by_date_range(csp_mig, dates_np, cur_lo, cur_hi, valid)
        prev_hi = ad_f - np.int64(n) * b_td
        prev_lo = prev_hi - np.int64(n - 1) * b_td
        prev_lo = np.maximum(prev_lo, t0)
        prev_hi = np.minimum(prev_hi, t1)
        prev = _sum_by_date_range(csp_mig, dates_np, prev_lo, prev_hi, valid)
        prev = np.where(np.isnan(prev), 0.0, prev)
        out[f"mig_sum_delta_{n}d_vs_prev_{n}d"] = np.where(valid, cur - prev, np.nan)

    pos = np.searchsorted(dates_np, ad_f, side="left")
    pos = np.clip(pos, 0, len(dates_np) - 1)
    for col in ["higgsfield", "comp_max", "hf_vs_comp_max", "hf_share_voice"]:
        if col not in gt.columns:
            continue
        gtv = gt[col].reindex(idx).values.astype(float)
        out[f"gt_{col}_at_sub"] = np.where(valid, gtv[pos], np.nan)

    s_hf = gt["higgsfield"].reindex(idx).values.astype(float)
    for n in ROLL_DAYS:
        pos_past = np.searchsorted(dates_np, ad_f - np.int64(n) * b_td, side="left")
        pos_past = np.clip(pos_past, 0, len(dates_np) - 1)
        out[f"gt_hf_delta_{n}d_at_sub"] = np.where(valid, s_hf[pos] - s_hf[pos_past], np.nan)

    for n in ROLL_DAYS:
        lo = ad_f - np.int64(n - 1) * b_td
        hi = ad_f
        lo = np.maximum(lo, t0)
        hi = np.minimum(hi, t1)
        out[f"oss_github_stars_sum_{n}d_at_sub"] = _sum_by_date_range(csp_gh, dates_np, lo, hi, valid)
        out[f"oss_hf_downloads_sum_{n}d_at_sub"] = _sum_by_date_range(csp_hf, dates_np, lo, hi, valid)

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build subscription-anchored migration/trend features.")
    ap.add_argument("--properties", type=Path, default=DATA / "train" / "train_users_properties.csv")
    ap.add_argument("--social", type=Path, default=DATA / "social_mentions.csv")
    ap.add_argument("--trends", type=Path, default=DATA / "google_trends.csv")
    ap.add_argument("--github", type=Path, default=DATA / "github_stars.csv")
    ap.add_argument("--hf", type=Path, default=DATA / "huggingface_downloads.csv")
    ap.add_argument("--out", type=Path, default=Path("output") / "train_user_market_anchor_features.csv")
    ap.add_argument(
        "--no-decode-subscription",
        action="store_true",
        help="Do not add DateOffset to subscription_start_date (use raw encoded timestamps).",
    )
    ap.add_argument(
        "--sub-shift-days",
        type=int,
        default=DEFAULT_SUB_SHIFT_DAYS,
        help="pd.DateOffset days added to subscription_start_date (default: 5, same as purchases script).",
    )
    ap.add_argument(
        "--sub-shift-months",
        type=int,
        default=DEFAULT_SUB_SHIFT_MONTHS,
        help="pd.DateOffset months (default: 0).",
    )
    ap.add_argument(
        "--sub-shift-years",
        type=int,
        default=DEFAULT_SUB_SHIFT_YEARS,
        help="pd.DateOffset years (default: 958, same as purchases script).",
    )
    args = ap.parse_args()

    gh = args.github if args.github.exists() else None
    hf = args.hf if args.hf.exists() else None

    df = build_user_table(
        args.properties,
        args.social,
        args.trends,
        gh,
        hf,
        sub_shift_days=args.sub_shift_days,
        sub_shift_months=args.sub_shift_months,
        sub_shift_years=args.sub_shift_years,
        decode_subscription=not args.no_decode_subscription,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_ok = int(df["sub_anchor_in_market_window"].sum())
    print(f"Wrote {len(df)} rows -> {args.out}")
    if not args.no_decode_subscription:
        print(
            f"Subscription decode: +{args.sub_shift_years}y +{args.sub_shift_months}m +{args.sub_shift_days}d "
            "(pd.DateOffset on UTC subscription_start_date)"
        )
    else:
        print("Subscription decode: OFF (raw timestamps)")
    print(f"Anchors inside market window [{ANCHOR_GRID_START.date()} .. {END_DATE.date()}]: {n_ok} / {len(df)}")
    if n_ok == 0:
        print(
            "WARNING: No subscription dates fall in the market window. "
            "Check subscription_start_date (synthetic years will not overlap YouTube 2025 data)."
        )


if __name__ == "__main__":
    main()
