import pandas as pd

folder = "data/preprocessed"
date_column = "purchase_time"
amount_column = "purchase_amount_dollars"
target_amount = 49

SHIFT_DAYS = 4
SHIFT_MONTHS = 4
SHIFT_YEARS = 958

for env in ["train", "test"]:
    path = f"{folder}/{env}/{env}_users_purchases.csv"
    df = pd.read_csv(path)
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    df[date_column] = df[date_column] + pd.DateOffset(days=SHIFT_DAYS, months=SHIFT_MONTHS, years=SHIFT_YEARS)

    filtered = df[df[amount_column] == target_amount].sort_values(date_column)

    if filtered.empty:
        print(f"[{env}] No ${target_amount} payments found")
        continue

    first = filtered.iloc[0]
    print(f"[{env}] First ${target_amount} payment: {first[date_column]}")
    print(f"  Total ${target_amount} payments: {len(filtered)}")
    print(f"  Date range: {filtered[date_column].min()} -> {filtered[date_column].max()}")
    print()