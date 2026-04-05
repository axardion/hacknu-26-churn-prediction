import csv
import json
import math
import time
import urllib.request
from collections import defaultdict

import pycountry

folder = "data/preprocessed"
countries_csv = f"{folder}/train/train_users_properties.csv"
payment_csv = f"{folder}/train/train_users.csv"
country_column = "country_code"
churn_column = "churn_status"
min_data_points_by_country = 0

def decode_country(code):
    try:
        return pycountry.countries.lookup(code).name
    except LookupError:
        return code

WB_REMAP = {
    "AQ": None, "TF": None, "RE": None, "EH": None, "SJ": None,
    "BV": None, "TK": None, "BQ": None, "NU": None, "JE": None, "GP": None,
    "CK": None, "MQ": None, "GG": None, "AI": None, "GI": None, "GF": None,
    "FK": None, "WF": None, "VA": None, "YT": None,
}

INDICATORS = {
    "gdp": "NY.GDP.MKTP.CD",
    "gdp_growth_pct": "NY.GDP.MKTP.KD.ZG",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "gdp_per_capita_growth_pct": "NY.GDP.PCAP.KD.ZG",
    "population": "SP.POP.TOTL",
}

def iso3_to_iso2(iso3):
    try:
        return pycountry.countries.get(alpha_3=iso3).alpha_2
    except (AttributeError, LookupError):
        return None

def fetch_wb(indicator, country_codes, start_year=2020, end_year=2025):
    codes_str = ";".join(country_codes)
    results = {}
    page = 1
    while True:
        url = (
            f"https://api.worldbank.org/v2/country/{codes_str}/indicator/{indicator}"
            f"?date={start_year}:{end_year}&format=json&per_page=500&page={page}"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Error fetching {indicator} page {page}: {e}")
            break
        if not raw or len(raw) < 2 or not raw[1]:
            print(f"  No data returned for {indicator} (page {page}), response: {str(raw)[:200]}")
            break
        if page == 1:
            print(f"  Sample entry keys: {list(raw[1][0].keys())}")
        for entry in raw[1]:
            iso3 = entry.get("countryiso3code", "")
            cc = iso3_to_iso2(iso3) if iso3 else None
            if not cc:
                cc = entry.get("country", {}).get("id", "").upper()
            if not cc or len(cc) != 2:
                continue
            year = int(entry["date"])
            val = entry["value"]
            if val is None:
                continue
            if cc not in results or year > results[cc][0]:
                results[cc] = (year, val)
        if page >= raw[0].get("pages", 1):
            break
        page += 1
        time.sleep(0.3)
    print(f"  {indicator}: got data for {len(results)} countries")
    return results

country_lookup = {}
with open(countries_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row.get("user_id") or row.get("id") or list(row.values())[0]
        country_lookup[uid] = row[country_column]

purchases_csv = "data/purchases_train.csv"
spend_by_country = defaultdict(list)
with open(purchases_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row.get("user_id") or row.get("id") or list(row.values())[0]
        country = country_lookup.get(uid)
        if country is None:
            continue
        try:
            spend_by_country[country].append(float(row["total_spend"]))
        except (ValueError, KeyError):
            continue

country_spend_stats = {}
for country, spends in spend_by_country.items():
    n = len(spends)
    avg = sum(spends) / n
    std = math.sqrt(sum((x - avg) ** 2 for x in spends) / (n - 1)) if n > 1 else 0
    country_spend_stats[country] = (round(avg, 2), round(std, 2))

stats = defaultdict(lambda: {"vol_churn": 0, "invol_churn": 0, "retained": 0})

with open(payment_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        uid = row.get("user_id") or row.get("id") or list(row.values())[0]
        status = row[churn_column]
        country = country_lookup.get(uid)
        if country is None:
            continue
        if status == "vol_churn":
            stats[country]["vol_churn"] += 1
        elif status == "invol_churn":
            stats[country]["invol_churn"] += 1
        elif status == "not_churned":
            stats[country]["retained"] += 1

unique_codes = list(dict.fromkeys(c.upper() for c in stats.keys()))
wb_codes = [WB_REMAP.get(c, c) for c in unique_codes if WB_REMAP.get(c, c) is not None]
wb_codes = [c for c in dict.fromkeys(wb_codes) if len(c) == 2 and c.isalpha()]

BATCH_SIZE = 50
all_econ = {ind: {} for ind in INDICATORS}

for i in range(0, len(wb_codes), BATCH_SIZE):
    batch = wb_codes[i:i + BATCH_SIZE]
    print(f"Fetching batch {i // BATCH_SIZE + 1} ({len(batch)} countries)...")
    for ind_name, ind_code in INDICATORS.items():
        result = fetch_wb(ind_code, batch)
        all_econ[ind_name].update(result)
        time.sleep(0.5)

print("\n--- Fetch summary ---")
for ind_name, ind_data in all_econ.items():
    print(f"  {ind_name}: {len(ind_data)} countries (sample: {dict(list(ind_data.items())[:3])})")

MANUAL_OVERRIDES = {
    "TW": {
        "gdp": (2023, 756590000000),
        "gdp_growth_pct": (2024, 4.59),
        "gdp_per_capita": (2023, 32339),
        "gdp_per_capita_growth_pct": (2024, 4.56),
        "population": (2024, 23500000),
    },
}
for cc, indicators in MANUAL_OVERRIDES.items():
    for ind_name, val in indicators.items():
        all_econ[ind_name][cc] = val

print()

TIKTOK_USERS = {
    "US": 136000000, "ID": 108000000, "BR": 91700000, "MX": 85400000, "PK": 66900000,
    "PH": 62300000, "RU": 56000000, "BD": 46500000, "EG": 41300000, "VN": 40900000,
    "TR": 40200000, "NG": 37400000, "IQ": 34300000, "SA": 34100000, "TH": 34000000,
    "CO": 32000000, "JP": 26900000, "GB": 24800000, "AR": 24400000, "PE": 24400000,
    "ZA": 23400000, "DE": 21800000, "FR": 21500000, "DZ": 21100000, "IT": 19800000,
    "MM": 19600000, "ES": 19000000, "UA": 17000000, "KZ": 15700000, "KE": 15100000,
    "MA": 14600000, "CL": 14600000, "VE": 13800000, "EC": 13500000, "CA": 12900000,
    "PL": 11400000, "AE": 11300000, "KH": 10700000, "GT": 10400000, "RO": 8500000,
    "TW": 8300000, "AU": 8000000, "BO": 7600000, "DO": 7200000, "KR": 7200000,
    "CD": 6700000, "AZ": 6700000, "BY": 6400000, "LK": 5800000, "NL": 5600000,
    "TN": 5200000, "LY": 5000000, "YE": 4400000, "HN": 4400000, "IL": 4200000,
    "LB": 4000000, "KW": 4000000, "GR": 3900000, "BE": 3700000, "PT": 3700000,
    "SV": 3700000, "SG": 3600000, "NI": 3400000, "CR": 3400000, "SE": 3300000,
    "HU": 3200000, "AO": 3100000, "SO": 3000000, "PA": 2900000, "MZ": 2800000,
    "RS": 2700000, "QA": 2600000, "AT": 2300000, "GE": 2300000, "JO": 2300000,
    "BG": 2300000, "HT": 2200000, "IE": 2200000, "CH": 2100000, "UY": 2100000,
    "CZ": 2000000, "GN": 2000000, "NZ": 1900000, "OM": 1800000, "NO": 1700000,
    "TD": 1700000, "MR": 1600000, "FI": 1600000, "JM": 1600000, "AL": 1400000,
    "DK": 1400000, "BH": 1200000, "MD": 1200000, "HR": 1100000, "SK": 1100000,
    "BA": 983000, "LT": 900000, "PY": 850000, "SN": 800000, "MN": 750000,
    "GH": 700000, "PS": 700000, "CM": 650000, "SI": 600000, "LV": 550000,
    "EE": 500000, "ME": 450000, "MK": 400000, "CY": 400000, "MT": 350000,
    "IS": 300000, "LU": 250000, "MU": 200000, "FJ": 180000, "MV": 150000,
    "BN": 140000, "BB": 100000, "SR": 90000, "BZ": 80000,
}

INSTAGRAM_USERS = {
    "IN": 392500000, "US": 172600000, "BR": 141400000, "ID": 90200000, "RU": 63900000,
    "TR": 58300000, "JP": 55500000, "MX": 48900000, "GB": 34700000, "DE": 33800000,
    "IT": 30700000, "FR": 29200000, "AR": 28400000, "ES": 27100000, "KR": 24200000,
    "PH": 22200000, "CO": 20900000, "CA": 19700000, "EG": 19400000, "IQ": 19200000,
    "TH": 19100000, "PK": 18600000, "SA": 17000000, "MY": 15700000, "AU": 14300000,
    "CL": 13100000, "NG": 12600000, "UA": 12600000, "MA": 12500000, "KZ": 12400000,
    "PL": 12300000, "DZ": 11800000, "VN": 10700000, "PE": 10300000, "UZ": 9400000,
    "VE": 9000000, "NL": 8900000, "AE": 7600000, "ZA": 7300000, "BD": 7200000,
    "EC": 6700000, "PT": 6500000, "SE": 6200000, "RO": 5800000, "BE": 5400000,
    "DO": 4900000, "CN": 4900000, "GR": 4700000, "IL": 4600000, "AZ": 4300000,
    "NP": 4100000, "CH": 4100000, "JO": 4000000, "CZ": 3900000, "BY": 3900000,
    "HK": 3800000, "TZ": 3700000, "AT": 3700000, "GT": 3500000, "TN": 3500000,
    "RS": 3300000, "KE": 3200000, "SG": 3200000, "KW": 3000000, "HU": 2900000,
    "KG": 2800000, "LK": 2600000, "PA": 2600000, "GH": 2500000, "CR": 2500000,
    "DK": 2400000, "IE": 2300000, "LB": 2300000, "NO": 2300000, "BG": 2200000,
    "BO": 2200000, "SV": 2100000, "FI": 2000000, "NZ": 2000000, "HR": 1900000,
    "UY": 1900000, "SK": 1800000, "GE": 1800000, "HN": 1700000, "PY": 1700000,
    "PS": 1600000, "QA": 1600000, "BA": 1400000, "LT": 1400000, "NI": 1300000,
    "OM": 1300000, "AL": 1300000, "MD": 1200000, "BH": 1200000, "JM": 1100000,
    "SI": 1000000, "LV": 900000, "MK": 900000, "CY": 900000, "EE": 700000,
    "ME": 600000, "MT": 500000, "IS": 400000, "LU": 400000, "MN": 400000,
    "MU": 350000, "TW": 9500000, "KH": 2300000, "MM": 3500000,
    "TT": 600000, "BN": 300000, "MV": 250000, "BB": 200000, "FJ": 250000,
    "SR": 200000, "BZ": 100000, "CM": 1500000, "SN": 1200000, "CI": 1000000,
}

LINKEDIN_USERS = {
    "US": 257000000, "IN": 161500000, "BR": 83200000, "GB": 47500000, "ID": 35800000,
    "CA": 28400000, "MX": 27500000, "PH": 20900000, "TR": 19400000, "AU": 17000000,
    "CO": 16500000, "ZA": 16100000, "AR": 16000000, "PK": 15800000, "EG": 13700000,
    "NG": 12100000, "SA": 11300000, "PE": 11300000, "BD": 10900000, "MY": 9800000,
    "AE": 9700000, "VN": 9500000, "CL": 9400000, "FR": 7900000, "RU": 7500000,
    "MA": 6600000, "TH": 6400000, "CN": 6300000, "UA": 6200000, "KE": 6100000,
    "VE": 5900000, "DZ": 5300000, "EC": 5300000, "JP": 5300000, "SG": 5000000,
    "KR": 5000000, "IT": 4100000, "TW": 4100000, "HK": 4000000, "DE": 4000000,
    "ES": 3500000, "GH": 3300000, "NZ": 3200000, "IL": 3000000, "NL": 3000000,
    "LK": 2700000, "TN": 2600000, "IQ": 2500000, "DO": 2200000, "NP": 2100000,
    "JO": 2000000, "CR": 2000000, "BO": 1900000, "GT": 1800000, "KZ": 1800000,
    "CI": 1700000, "UG": 1700000, "PL": 1700000, "QA": 1700000, "RS": 1700000,
    "UY": 1600000, "ET": 1600000, "SE": 1600000, "TZ": 1600000, "CM": 1500000,
    "PA": 1500000, "SN": 1400000, "LB": 1300000, "AO": 1300000, "BE": 1300000,
    "KW": 1200000, "CH": 1200000, "PR": 1200000, "OM": 1100000, "ZW": 1100000,
    "PY": 1100000, "AZ": 1100000, "PT": 1100000, "SV": 1100000, "MM": 1100000,
    "BY": 1100000, "ZM": 1100000,
}

X_USERS = {
    "US": 105100000, "JP": 74500000, "ID": 23600000, "PL": 23300000, "IN": 23100000,
    "GB": 19300000, "TR": 18500000, "DE": 17400000, "MX": 16600000, "SA": 15700000,
    "TH": 13600000, "HK": 12800000, "FR": 12600000, "KR": 10800000, "CA": 10300000,
    "ES": 9800000, "NL": 8700000, "PH": 8200000, "SG": 8100000, "FI": 7700000,
    "VN": 7500000, "NG": 7300000, "AR": 7000000, "TW": 6300000, "CN": 5300000,
    "CO": 5000000, "MY": 5000000, "IT": 5000000, "AU": 4800000, "EG": 4500000,
    "ZA": 3100000, "CL": 3100000, "PK": 3000000, "AE": 3000000, "IQ": 2800000,
    "CH": 2200000, "KE": 2100000, "PE": 2100000, "SE": 1900000, "BR": 1800000,
    "RO": 1700000, "EC": 1700000, "BE": 1700000, "PT": 1600000, "FM": 1600000,
    "UA": 1500000, "IE": 1500000, "KW": 1400000, "YE": 1400000, "AT": 1400000,
    "BD": 1100000, "GH": 1100000, "DZ": 1100000, "CZ": 1100000, "GR": 1100000,
    "GT": 1000000, "MA": 988400, "NO": 966600, "IL": 955900, "OM": 943400,
    "DK": 929900, "JO": 820500, "UG": 800500, "BG": 766900, "NZ": 715600,
    "VE": 698500, "HU": 696100, "KH": 690300, "DO": 679800, "RS": 677800,
    "QA": 666500, "SV": 641200, "UY": 614800, "PR": 612800, "PY": 572400,
    "CR": 571500, "LY": 550400, "LB": 533500, "PA": 516500, "RU": 506400,
    "BO": 501600,
}

data = []
econ_keys = set(all_econ["gdp"].keys())
print(f"Econ data available for: {sorted(econ_keys)[:20]}...")
print(f"Sample econ values: {dict(list(all_econ['gdp'].items())[:5])}")

for c, s in stats.items():
    total = s["vol_churn"] + s["invol_churn"] + s["retained"]
    if total < min_data_points_by_country:
        continue
    vol_pct = round(s["vol_churn"] / total * 100, 2) if total else 0

    wb = WB_REMAP.get(c.upper(), c.upper())
    if c in ["JP", "US", "DE"]:
        print(f"  Debug {c}: wb={wb}, in econ={wb in econ_keys}, gdp={all_econ['gdp'].get(wb)}")

    row = {
        "country_code": c,
        "country_name": decode_country(c),
        "vol_churn": s["vol_churn"],
        "invol_churn": s["invol_churn"],
        "retained": s["retained"],
        "vol_churn_pct": vol_pct,
        "invol_churn_pct": round(s["invol_churn"] / total * 100, 2) if total else 0,
        "retained_pct": round(s["retained"] / total * 100, 2) if total else 0,
        "_sort": vol_pct * math.sqrt(s["vol_churn"]),
    }

    if wb is None:
        row.update({"gdp": "", "gdp_growth_pct": "", "gdp_per_capita": "", "gdp_per_capita_growth_pct": "", "population": "", "data_year": ""})
    else:
        gdp_info = all_econ["gdp"].get(wb)
        row["data_year"] = gdp_info[0] if gdp_info else ""
        row["gdp"] = gdp_info[1] if gdp_info and gdp_info[1] is not None else ""
        for key in ["gdp_growth_pct", "gdp_per_capita", "gdp_per_capita_growth_pct"]:
            info = all_econ[key].get(wb)
            row[key] = round(info[1], 2) if info and info[1] is not None else ""
        pop = all_econ["population"].get(wb)
        row["population"] = int(pop[1]) if pop and pop[1] is not None else ""

    cc_upper = c.upper()
    row["tiktok_users"] = TIKTOK_USERS.get(cc_upper, "")
    row["instagram_users"] = INSTAGRAM_USERS.get(cc_upper, "")
    row["linkedin_users"] = LINKEDIN_USERS.get(cc_upper, "")
    row["x_users"] = X_USERS.get(cc_upper, "")

    spend = country_spend_stats.get(c)
    row["avg_total_spend"] = spend[0] if spend else ""
    row["std_total_spend"] = spend[1] if spend else ""

    data.append(row)

data.sort(key=lambda r: r["_sort"], reverse=True)

for r in data:
    if r["country_code"] in ["JP", "US", "DE"]:
        print(f"  Row {r['country_code']}: gdp={r.get('gdp')}, pop={r.get('population')}, growth={r.get('gdp_growth_pct')}")
        break

output = "data/countries.csv"
import os
os.makedirs("output", exist_ok=True)
fieldnames = [
    "country_code", "country_name",
    "vol_churn", "invol_churn", "retained",
    "vol_churn_pct", "invol_churn_pct", "retained_pct",
    "population", "gdp", "gdp_growth_pct", "gdp_per_capita", "gdp_per_capita_growth_pct", "data_year",
    "tiktok_users", "instagram_users", "linkedin_users", "x_users",
    "avg_total_spend", "std_total_spend",
]
with open(output, newline="", mode="w", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(data)

print(f"Written {len(data)} rows to {output}")