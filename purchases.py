import csv
from collections import defaultdict
from datetime import timedelta
from dateutil import parser as dtparser
import os

TIME_SHIFT = timedelta(days=958 * 365 + 958 // 4 - 958 // 100 + 958 // 400 + 6)


def read_purchases(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_user_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return {r["user_id"] for r in csv.DictReader(f)}


def aggregate(rows, categories):
    users = defaultdict(lambda: {"amounts": [], "cat_amounts": defaultdict(list), "shifted_times": []})
    for r in rows:
        uid = r["user_id"]
        amt = float(r["purchase_amount_dollars"])
        cat = r["purchase_type"]
        users[uid]["amounts"].append(amt)
        users[uid]["cat_amounts"][cat].append(amt)
        shifted = dtparser.parse(r["purchase_time"]) + TIME_SHIFT
        users[uid]["shifted_times"].append(shifted)
    return users


def build_header(categories):
    return [
        "user_id",
        "total_spend",
        *[f"spend_{c}" for c in categories],
        *[f"has_{c}" for c in categories],
        *[f"count_{c}" for c in categories],
        "highest_single_spend",
        "lowest_single_spend",
        "mean_spend",
        "total_number_of_transactions",
        "days_between_first_last_purchase",
    ]


def write_csv(path, users, categories, all_user_ids):
    header = build_header(categories)
    missing_ids = sorted(all_user_ids - set(users.keys()))
    zero_row_len = len(header) - 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for uid, d in sorted(users.items()):
            amounts = d["amounts"]
            cat_amounts = d["cat_amounts"]
            writer.writerow([
                uid,
                round(sum(amounts), 2),
                *[round(sum(cat_amounts.get(c, [])), 2) for c in categories],
                *[int(bool(cat_amounts.get(c))) for c in categories],
                *[len(cat_amounts.get(c, [])) for c in categories],
                round(max(amounts), 2),
                round(min(amounts), 2),
                round(sum(amounts) / len(amounts), 2),
                len(amounts),
                (max(d["shifted_times"]) - min(d["shifted_times"])).days if len(amounts) > 1 else 0,
            ])
        for uid in missing_ids:
            writer.writerow([uid] + [0] * zero_row_len)
    print(f"Written {len(users)} users + {len(missing_ids)} zero-filled → {path}")


os.makedirs("data", exist_ok=True)

train_rows = read_purchases("data/preprocessed/train/train_users_purchases.csv")
test_rows = read_purchases("data/preprocessed/test/test_users_purchases.csv")

all_rows = train_rows + test_rows
categories = sorted({r["purchase_type"] for r in all_rows})

train_users = aggregate(train_rows, categories)
train_user_ids = read_user_ids("data/preprocessed/train/train_users.csv")
write_csv("data/purchases_train.csv", train_users, categories, train_user_ids)

all_users = aggregate(all_rows, categories)
all_user_ids = train_user_ids | read_user_ids("data/preprocessed/test/test_users.csv")
write_csv("data/purchases_test.csv", all_users, categories, all_user_ids)