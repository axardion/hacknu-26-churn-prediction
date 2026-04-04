import csv

main_train_file = "data/preprocessed/train/train_users.csv"
auxiliary_train_file = "data/preprocessed/train/train_users_properties.csv"
main_test_file = "data/preprocessed/test/test_users.csv"
auxiliary_test_file = "data/preprocessed/test/test_users_properties.csv"

train_countries_file = "data/countries.csv"
test_countries_file = "data/countries.csv"

user_id_column = "user_id"
country_code_column = "country_code"
gdp_per_capita_column = "gpd_per_capita"
tiktok_users_column = "tiktok_users"
population_column = "population"
result_column = "gdp_per_tiktok_user_scaled"


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


def build_country_metric_map(country_rows):
    result = {}
    for row in country_rows:
        try:
            result[row[country_code_column]] = (
                float(row[gdp_per_capita_column]) / float(row[tiktok_users_column]) * float(row[population_column])
            )
        except (ValueError, ZeroDivisionError, KeyError):
            result[row[country_code_column]] = ""
    return result


def enrich_and_write(main_path, auxiliary_path, countries_path):
    main_rows = load_csv(main_path)
    auxiliary_rows = load_csv(auxiliary_path)
    country_rows = load_csv(countries_path)

    user_country = build_user_country_map(auxiliary_rows)
    country_metric = build_country_metric_map(country_rows)

    fieldnames = list(main_rows[0].keys()) + [result_column]

    for row in main_rows:
        uid = row[user_id_column]
        cc = user_country.get(uid, "")
        row[result_column] = country_metric.get(cc, "")

    with open(main_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(main_rows)


enrich_and_write(main_train_file, auxiliary_train_file, train_countries_file)
enrich_and_write(main_test_file, auxiliary_test_file, test_countries_file)