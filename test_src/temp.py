import csv

with open("data/preprocessed/train/train_users_quizzes.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)
    rows = list(reader)

bench = 4
count_bench = 0
count_total = 0
for row in rows:
    if row.count("skipped") >= bench:
        count_bench += 1
        count_total += 1
    else:
        count_total += 1

print(count_bench / count_total * 100)