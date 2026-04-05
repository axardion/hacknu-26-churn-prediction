import pandas as pd

folder = "data/preprocessed"
date_column = "purchase_time"

SHIFT_DAYS = 4
SHIFT_MONTHS = 4
SHIFT_YEARS = 958

path = f"{folder}/train/train_users_purchases.csv"
df = pd.read_csv(path)
df[date_column] = pd.to_datetime(df[date_column], utc=True)
df[date_column] = df[date_column] + pd.DateOffset(days=SHIFT_DAYS, months=SHIFT_MONTHS, years=SHIFT_YEARS)

print(f"Test set date range: {df[date_column].min()} -> {df[date_column].max()}")