import csv

with open('data/preprocessed/train/train_users_purchases.csv', "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)
    rows = list(reader)

for user