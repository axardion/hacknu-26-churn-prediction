import csv
import math

main_train_file = "data/preprocessed/train/train_users.csv"
main_test_file = "data/preprocessed/test/test_users.csv"
auxiliary_train_file = "data/preprocessed/train/train_users_properties.csv"
auxiliary_test_file = "data/preprocessed/test/test_users_properties.csv"
purchases_train_file = "data/purchases_train.csv"
purchases_test_file = "data/purchases_test.csv"
countries_file = "data/countries.csv"

user_id_column = "user_id"
country_code_column = "country_code"
gdp_column = "gdp_per_capita"
spend_column = "highest_single_spend"
new_feature = "highest_single_spend_D_exp_gdp_per_capita"


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_user_country_map(auxiliary_rows):
    mapping = {}
    for row in auxiliary_rows:
        uid = row[user_id_column]
        if uid not in mapping and country_code_column in row:
            mapping[uid] = row[country_code_column]
    return mapping


def build_country_gdp_map(country_rows):
    return {row[country_code_column]: row[gdp_column] for row in country_rows}


def build_user_spend_map(purchase_rows):
    mapping = {}
    for row in purchase_rows:
        uid = row[user_id_column]
        if uid not in mapping and spend_column in row:
            mapping[uid] = row[spend_column]
    return mapping


def enrich_and_write(main_path, auxiliary_path, purchases_path, countries_path):
    main_rows = load_csv(main_path)
    auxiliary_rows = load_csv(auxiliary_path)
    purchase_rows = load_csv(purchases_path)
    country_rows = load_csv(countries_path)

    user_country = build_user_country_map(auxiliary_rows)
    country_gdp = build_country_gdp_map(country_rows)
    user_spend = build_user_spend_map(purchase_rows)

    fieldnames = list(main_rows[0].keys()) + [new_feature]

    for row in main_rows:
        uid = row[user_id_column]
        cc = user_country.get(uid, "")
        gdp_val = country_gdp.get(cc, "")
        spend_val = user_spend.get(uid, "")

        if gdp_val and spend_val:
            try:
                row[new_feature] = float(spend_val) / math.exp(float(gdp_val))
            except (ValueError, OverflowError):
                row[new_feature] = ""
        else:
            row[new_feature] = ""

    with open(main_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(main_rows)


enrich_and_write(main_train_file, auxiliary_train_file, purchases_train_file, countries_file)
enrich_and_write(main_test_file, auxiliary_test_file, purchases_test_file, countries_file)