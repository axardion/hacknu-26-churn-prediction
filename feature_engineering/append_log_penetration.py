import csv
import math

main_train_file = "data/preprocessed/train/train_users.csv"
auxiliary_train_file = "data/preprocessed/train/train_users_properties.csv"
main_test_file = "data/preprocessed/test/test_users.csv"
auxiliary_test_file = "data/preprocessed/test/test_users_properties.csv"

train_countries_file = "data/countries.csv"
test_countries_file = "data/countries.csv"

user_id_column = "user_id"
country_code_column = "country_code"
churn_density_column = "log_churn_density"


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


def build_country_churn_density_map(country_rows):
    result = {}
    for row in country_rows:
        cc = row[country_code_column]
        total_churn = float(row["vol_churn"]) + float(row["invol_churn"]) + float(row["retained"])
        population = float(row["population"])
        if population > 0 and total_churn > 0:
            result[cc] = math.log(total_churn / population)
        else:
            result[cc] = ""
    return result


def enrich_and_write(main_path, auxiliary_path, countries_path):
    main_rows = load_csv(main_path)
    auxiliary_rows = load_csv(auxiliary_path)
    country_rows = load_csv(countries_path)

    user_country = build_user_country_map(auxiliary_rows)
    country_churn = build_country_churn_density_map(country_rows)

    fieldnames = list(main_rows[0].keys()) + [churn_density_column]

    for row in main_rows:
        uid = row[user_id_column]
        cc = user_country.get(uid, "")
        row[churn_density_column] = country_churn.get(cc, "")

    with open(main_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(main_rows)


enrich_and_write(main_train_file, auxiliary_train_file, train_countries_file)
enrich_and_write(main_test_file, auxiliary_test_file, test_countries_file)