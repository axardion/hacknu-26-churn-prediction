import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from itertools import product

folder = "data/preprocessed"
env1 = "train"
file_path_1 = f"{folder}/{env1}/{env1}_users_purchases.csv"
SHIFT_DAYS_LIST_1 = [6]
SHIFT_MONTHS_LIST_1 = [0]
SHIFT_YEARS_LIST_1 = [958]

env2 = "skip_this_file"
file_path_2 = f"{folder}/{env2}/{env2}_users_purchases.csv"
SHIFT_DAYS_LIST_2 = [6]
SHIFT_MONTHS_LIST_2 = [0]
SHIFT_YEARS_LIST_2 = [958]

date_column = "purchase_time"
amount_column = "purchase_amount_dollars"

os.makedirs("output", exist_ok=True)

sources = []

if env1 != "skip_this_file":
    df1_raw = pd.read_csv(file_path_1)
    df1_raw[date_column] = pd.to_datetime(df1_raw[date_column], utc=True)
    perms1 = list(product(SHIFT_DAYS_LIST_1, SHIFT_MONTHS_LIST_1, SHIFT_YEARS_LIST_1))
    sources.append(("f1", df1_raw, perms1))
else:
    sources.append(("f1", None, [(0, 0, 0)]))

if env2 != "skip_this_file":
    df2_raw = pd.read_csv(file_path_2)
    df2_raw[date_column] = pd.to_datetime(df2_raw[date_column], utc=True)
    perms2 = list(product(SHIFT_DAYS_LIST_2, SHIFT_MONTHS_LIST_2, SHIFT_YEARS_LIST_2))
    sources.append(("f2", df2_raw, perms2))
else:
    sources.append(("f2", None, [(0, 0, 0)]))

all_combos = list(product(
    sources[0][2] if sources[0][1] is not None else [(0, 0, 0)],
    sources[1][2] if sources[1][1] is not None else [(0, 0, 0)],
))

for (d1, mo1, y1), (d2, mo2, y2) in all_combos:
    frames = []

    if sources[0][1] is not None:
        df1 = sources[0][1].copy()
        df1[date_column] = df1[date_column] + pd.DateOffset(days=d1, months=mo1, years=y1)
        frames.append(df1)

    if sources[1][1] is not None:
        df2 = sources[1][1].copy()
        df2[date_column] = df2[date_column] + pd.DateOffset(days=d2, months=mo2, years=y2)
        frames.append(df2)

    df = pd.concat(frames, ignore_index=True)
    df["month"] = df[date_column].dt.to_period("M")

    # --- Added: Print overall min/max date ---
    start_date = df[date_column].min()
    end_date = df[date_column].max()
    print(f"Modified Data Time Period: {start_date} to {end_date}")
    # -----------------------------------------

    tag = f"f1_{d1}d_{mo1}m_{y1}y"
    if sources[1][1] is not None:
        tag += f"__f2_{d2}d_{mo2}m_{y2}y"
    output_path = f"output/revenue/revenue_dynamics_{tag}.png"

    monthly_rev = df.groupby("month")[amount_column].sum().sort_index()
    monthly_min = df.groupby("month")[date_column].min().sort_index()
    monthly_max = df.groupby("month")[date_column].max().sort_index()

    day_counts = []
    normalized = []
    for m in monthly_rev.index:
        actual_days = (monthly_max[m] - monthly_min[m]).days + 1
        day_counts.append(actual_days)
        normalized.append(monthly_rev[m] / actual_days * 30)

    normalized = pd.Series(normalized, index=monthly_rev.index)

    x = monthly_rev.index.astype(str)
    step = max(1, len(x) // 20)

    shift_label = f"F1({d1}d,{mo1}m,{y1}y)"
    if sources[1][1] is not None:
        shift_label += f" + F2({d2}d,{mo2}m,{y2}y)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7))

    bars1 = ax1.bar(x, monthly_rev.values, color="#4C72B0", edgecolor="white", linewidth=0.3)
    for bar, days in zip(bars1, day_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, str(days),
                 ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Revenue ($)", fontsize=12)
    ax1.set_title(f"Raw Revenue per Month | {shift_label}", fontsize=12, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.set_xticks(range(0, len(x), step))
    ax1.set_xticklabels(x[::step], rotation=45, ha="right", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bars2 = ax2.bar(x, normalized.values, color="#DD8452", edgecolor="white", linewidth=0.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, "30",
                 ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Revenue ($)", fontsize=12)
    ax2.set_title(f"Normalized Revenue (30 days) | {shift_label}", fontsize=12, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.set_xticks(range(0, len(x), step))
    ax2.set_xticklabels(x[::step], rotation=45, ha="right", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved to {output_path} | {shift_label} | months: {len(monthly_rev)} | total: ${monthly_rev.sum():,.2f}")

    print(f"\n{'Month':<12} {'#1 Price':>10} {'#1 %':>7} {'#2 Price':>10} {'#2 %':>7} {'#3 Price':>10} {'#3 %':>7} {'Total':>8}")
    print("-" * 76)
    for m in monthly_rev.index:
        month_df = df[df["month"] == m]
        total = len(month_df)
        counts = month_df[amount_column].value_counts()
        top3 = counts.head(3)
        p1, c1 = top3.index[0], top3.iloc[0]
        if len(top3) > 1:
            p2, c2 = top3.index[1], top3.iloc[1]
        else:
            p2, c2 = None, 0
        if len(top3) > 2:
            p3, c3 = top3.index[2], top3.iloc[2]
        else:
            p3, c3 = None, 0
        pct1 = c1 / total * 100
        pct2 = c2 / total * 100 if c2 else 0
        pct3 = c3 / total * 100 if c3 else 0
        p2_str = f"${p2:>9.2f}" if p2 is not None else f"{'N/A':>10}"
        p3_str = f"${p3:>9.2f}" if p3 is not None else f"{'N/A':>10}"
        print(f"{str(m):<12} ${p1:>9.2f} {pct1:>6.1f}% {p2_str} {pct2:>6.1f}% {p3_str} {pct3:>6.1f}% {total:>8}")
    print()