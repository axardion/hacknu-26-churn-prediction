from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# countries info
FORMULAS_B = [
    "18",
    "19",
]

# purchases.csv columns (0-indexed):
# 0:user_id 1:total_spend 2:spend_Credits_package 3:spend_Gift
# 4:spend_Reactivation 5:spend_Subscription_Create 6:spend_Subscription_Update
# 7:spend_Upsell 8:has_Credits_package 9:has_Gift 10:has_Reactivation
# 11:has_Subscription_Create 12:has_Subscription_Update 13:has_Upsell
# 14:count_Credits_package 15:count_Gift 16:count_Reactivation
# 17:count_Subscription_Create 18:count_Subscription_Update 19:count_Upsell
# 20:highest_single_spend 21:lowest_single_spend 22:mean_spend
# 23:total_number_of_transactions
FORMULAS_C = [
    "23",
]

# subscription info
FORMULAS_D = [
    "2",
    "3",
]

# duration info
FORMULAS_E = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
]

INTERACTION_FORMULAS = [
    "d_0 * e_0",
    "d_0 * e_1",
    "d_0 * e_2",
    "d_0 * e_3",
    "d_0 * e_4",
    "d_0 * e_5",
    "d_0 / e_0",
    "d_0 / e_1",
    "d_0 / e_2",
    "d_0 / e_3",
    "d_0 / e_4",
    "d_0 / e_5",
]


def tokenize(formula):
    tokens = []
    i = 0
    s = formula.replace(" ", "")
    while i < len(s):
        if s[i].isdigit() or (s[i] == '.' and i + 1 < len(s) and s[i + 1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(("NUM", s[i:j]))
            i = j
        elif s[i:i+4] == "sqrt":
            tokens.append(("FUNC", "sqrt"))
            i += 4
        elif s[i:i+3] == "exp":
            tokens.append(("FUNC", "exp"))
            i += 3
        elif s[i:i+3] == "abs":
            tokens.append(("FUNC", "abs"))
            i += 3
        elif s[i:i+3] == "log":
            tokens.append(("FUNC", "log"))
            i += 3
        elif s[i:i+3] == "pow":
            tokens.append(("FUNC", "pow"))
            i += 3
        elif s[i] in "+-*/^":
            tokens.append(("OP", s[i]))
            i += 1
        elif s[i] == '(':
            tokens.append(("LPAREN", "("))
            i += 1
        elif s[i] == ')':
            tokens.append(("RPAREN", ")"))
            i += 1
        elif s[i] == ',':
            tokens.append(("COMMA", ","))
            i += 1
        else:
            raise ValueError(f"Unknown character: {s[i]}")
    return tokens

def parse_expr(tokens, pos):
    left = parse_term(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]][0] == "OP" and tokens[pos[0]][1] in "+-":
        op = tokens[pos[0]][1]
        pos[0] += 1
        right = parse_term(tokens, pos)
        left = ("add" if op == "+" else "sub", left, right)
    return left

def parse_term(tokens, pos):
    left = parse_power(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]][0] == "OP" and tokens[pos[0]][1] in "*/":
        op = tokens[pos[0]][1]
        pos[0] += 1
        right = parse_power(tokens, pos)
        left = ("mul" if op == "*" else "div", left, right)
    return left

def parse_power(tokens, pos):
    base = parse_unary(tokens, pos)
    if pos[0] < len(tokens) and tokens[pos[0]] == ("OP", "^"):
        pos[0] += 1
        exp = parse_unary(tokens, pos)
        return ("pow", base, exp)
    return base

def parse_unary(tokens, pos):
    if pos[0] < len(tokens) and tokens[pos[0]] == ("OP", "-"):
        pos[0] += 1
        operand = parse_atom(tokens, pos)
        return ("neg", operand)
    return parse_atom(tokens, pos)

def parse_atom(tokens, pos):
    if pos[0] >= len(tokens):
        raise ValueError("Unexpected end of formula")
    tok = tokens[pos[0]]
    if tok[0] == "FUNC":
        fname = tok[1]
        pos[0] += 1
        if pos[0] >= len(tokens) or tokens[pos[0]] != ("LPAREN", "("):
            raise ValueError(f"Expected '(' after {fname}")
        pos[0] += 1
        args = [parse_expr(tokens, pos)]
        while pos[0] < len(tokens) and tokens[pos[0]] == ("COMMA", ","):
            pos[0] += 1
            args.append(parse_expr(tokens, pos))
        if pos[0] >= len(tokens) or tokens[pos[0]] != ("RPAREN", ")"):
            raise ValueError(f"Expected ')' after {fname} arguments")
        pos[0] += 1
        return ("func", fname, args)
    if tok == ("LPAREN", "("):
        pos[0] += 1
        node = parse_expr(tokens, pos)
        if pos[0] >= len(tokens) or tokens[pos[0]] != ("RPAREN", ")"):
            raise ValueError("Missing closing ')'")
        pos[0] += 1
        return node
    if tok[0] == "NUM":
        pos[0] += 1
        if '.' not in tok[1]:
            return ("col", int(tok[1]))
        else:
            return ("const", float(tok[1]))
    raise ValueError(f"Unexpected token: {tok}")

def parse_formula(formula):
    tokens = tokenize(formula)
    pos = [0]
    result = parse_expr(tokens, pos)
    if pos[0] != len(tokens):
        raise ValueError(f"Unexpected token at position {pos[0]}: {tokens[pos[0]:]}")
    return result

def evaluate(node, row):
    kind = node[0]
    if kind == "col":
        return float(row[node[1]])
    if kind == "const":
        return node[1]
    if kind == "neg":
        return -evaluate(node[1], row)
    if kind == "add":
        return evaluate(node[1], row) + evaluate(node[2], row)
    if kind == "sub":
        return evaluate(node[1], row) - evaluate(node[2], row)
    if kind == "mul":
        return evaluate(node[1], row) * evaluate(node[2], row)
    if kind == "div":
        return evaluate(node[1], row) / evaluate(node[2], row)
    if kind == "pow":
        return evaluate(node[1], row) ** evaluate(node[2], row)
    if kind == "func":
        fname = node[1]
        args = [evaluate(a, row) for a in node[2]]
        if fname == "sqrt":
            return math.sqrt(args[0])
        if fname == "exp":
            return math.exp(args[0])
        if fname == "abs":
            return abs(args[0])
        if fname == "log":
            return math.log(args[0]) if len(args) == 1 else math.log(args[0], args[1])
        if fname == "pow":
            return args[0] ** args[1]
    raise ValueError(f"Unknown node: {node}")

def describe(formula, hdr):
    def replace_col(m):
        idx = int(m.group())
        return f'"{hdr[idx]}"' if idx < len(hdr) else f"?{idx}"
    return re.sub(r'(?<![.\d])\b(\d+)\b(?!\.\d)', lambda m: replace_col(m), formula)

def pearson(va, vb):
    if len(va) < 2:
        return float("nan")
    a = np.array(va, dtype=np.float64)
    b = np.array(vb, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    cc = np.corrcoef(a[mask], b[mask])
    return float(cc[0, 1])


def read_users(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    c0 = df.columns[0]
    if str(c0).startswith("Unnamed") or str(c0).lower() == "skipped":
        df = df.drop(columns=c0)
    df["churn"] = df["churn_status"].isin(["vol_churn", "invol_churn"]).astype(np.int8)
    df["vol_vs_invol"] = np.where(
        df["churn_status"] == "vol_churn", 1,
        np.where(df["churn_status"] == "invol_churn", 0, np.nan),
    )
    return df[["user_id", "churn", "vol_vs_invol"]]


def build_country_features(props_path, countries_path):
    props = pd.read_csv(props_path, low_memory=False, dtype={"user_id": str})
    countries = pd.read_csv(countries_path, low_memory=False)
    countries_header = list(countries.columns)

    trees_b = [parse_formula(f) for f in FORMULAS_B]
    labels_b = [describe(f, countries_header) for f in FORMULAS_B]

    country_vals: dict[str, list[float | None]] = {}
    for _, r in countries.iterrows():
        crow = [str(v) for v in r.values]
        code = crow[0]
        vals = []
        for tb in trees_b:
            try:
                v = evaluate(tb, crow)
                vals.append(v if math.isfinite(v) else None)
            except Exception:
                vals.append(None)
        country_vals[code] = vals

    feat_names = [f"country_B{j}" for j in range(len(FORMULAS_B))]
    records = []
    for _, r in props.iterrows():
        cc = r.get("country_code")
        if pd.isna(cc):
            continue
        vals = country_vals.get(str(cc))
        if vals is None:
            continue
        rec = {"user_id": str(r["user_id"])}
        for k, fn in enumerate(feat_names):
            rec[fn] = vals[k]
        records.append(rec)

    return pd.DataFrame(records), feat_names, labels_b


def build_purchase_features(purchases_path):
    purchases = pd.read_csv(purchases_path, low_memory=False, dtype={"user_id": str})
    purchases_header = list(purchases.columns)

    trees_c = [parse_formula(f) for f in FORMULAS_C]
    labels_c = [describe(f, purchases_header) for f in FORMULAS_C]

    feat_names = [f"purchase_C{j}" for j in range(len(FORMULAS_C))]
    records = []
    for _, r in purchases.iterrows():
        crow = [str(v) for v in r.values]
        rec = {"user_id": str(r["user_id"])}
        for k, (tc, fn) in enumerate(zip(trees_c, feat_names)):
            try:
                v = evaluate(tc, crow)
                rec[fn] = v if math.isfinite(v) else None
            except Exception:
                rec[fn] = None
        records.append(rec)

    return pd.DataFrame(records), feat_names, labels_c


def build_subscription_features(subscriptions_path):
    subscriptions = pd.read_csv(subscriptions_path, low_memory=False, dtype={"user_id": str})
    subscriptions_header = list(subscriptions.columns)

    trees_d = [parse_formula(f) for f in FORMULAS_D]
    labels_d = [describe(f, subscriptions_header) for f in FORMULAS_D]

    feat_names = [f"subscription_D{j}" for j in range(len(FORMULAS_D))]
    records = []
    for _, r in subscriptions.iterrows():
        crow = [str(v) for v in r.values]
        rec = {"user_id": str(r["user_id"])}
        for k, (td, fn) in enumerate(zip(trees_d, feat_names)):
            try:
                v = evaluate(td, crow)
                rec[fn] = v if math.isfinite(v) else None
            except Exception:
                rec[fn] = None
        records.append(rec)

    return pd.DataFrame(records), feat_names, labels_d


def build_duration_features(durations_path):
    durations = pd.read_csv(durations_path, low_memory=False, dtype={"user_id": str})
    durations_header = list(durations.columns)

    trees_e = [parse_formula(f) for f in FORMULAS_E]
    labels_e = [describe(f, durations_header) for f in FORMULAS_E]

    feat_names = [f"duration_E{j}" for j in range(len(FORMULAS_E))]
    records = []
    for _, r in durations.iterrows():
        crow = [str(v) for v in r.values]
        rec = {"user_id": str(r["user_id"])}
        for k, (te, fn) in enumerate(zip(trees_e, feat_names)):
            try:
                v = evaluate(te, crow)
                rec[fn] = v if math.isfinite(v) else None
            except Exception:
                rec[fn] = None
        records.append(rec)

    return pd.DataFrame(records), feat_names, labels_e


def build_interaction_features(df, b_feat_names, c_feat_names, d_feat_names, e_feat_names, b_labels, c_labels, d_labels, e_labels):
    feat_names = []
    labels = []
    all_feat_map = {
        "b": (b_feat_names, b_labels, "B"),
        "c": (c_feat_names, c_labels, "C"),
        "d": (d_feat_names, d_labels, "D"),
        "e": (e_feat_names, e_labels, "E"),
    }
    for idx, formula in enumerate(INTERACTION_FORMULAS):
        refs = re.findall(r'[bcde]_\d+', formula)
        ref_to_slot: dict[str, int] = {}
        slot_to_col: dict[int, str] = {}
        for ref in refs:
            if ref in ref_to_slot:
                continue
            kind, fidx = ref[0], int(ref[2:])
            feat_list, _, tag = all_feat_map[kind]
            if fidx >= len(feat_list):
                raise IndexError(f"{kind}_{fidx} out of range, only {len(feat_list)} {tag} features (0..{len(feat_list)-1})")
            col = feat_list[fidx]
            slot = len(ref_to_slot)
            ref_to_slot[ref] = slot
            slot_to_col[slot] = col

        numeric_formula = formula
        for ref in sorted(ref_to_slot, key=len, reverse=True):
            numeric_formula = numeric_formula.replace(ref, str(ref_to_slot[ref]))

        tree = parse_formula(numeric_formula)

        label_formula = formula
        for ref in sorted(ref_to_slot, key=len, reverse=True):
            kind, fidx = ref[0], int(ref[2:])
            _, lbl_list, tag = all_feat_map[kind]
            label_formula = label_formula.replace(ref, f"{tag}({lbl_list[fidx]})")

        fn = f"interact_I{idx}"
        feat_names.append(fn)
        labels.append(label_formula)

        col_list = [slot_to_col[s] for s in range(len(slot_to_col))]
        vals = []
        for _, r in df[col_list].iterrows():
            row = [str(v) for v in r.values]
            try:
                v = evaluate(tree, row)
                vals.append(v if math.isfinite(v) else None)
            except Exception:
                vals.append(None)
        df[fn] = vals

    return feat_names, labels

def temporal_subset_last_fraction(
    df: pd.DataFrame,
    props_path: Path,
    fraction: float = 0.7,
    date_col: str = "subscription_start_date",
) -> pd.DataFrame:
    props = pd.read_csv(props_path, low_memory=False, dtype={"user_id": str})
    props[date_col] = pd.to_datetime(props[date_col], utc=True, errors="coerce")
    dates = props[["user_id", date_col]].dropna(subset=[date_col])
    dates = dates.sort_values(date_col)
    cutoff_idx = int(len(dates) * (1 - fraction))
    recent_users = set(dates.iloc[cutoff_idx:]["user_id"])
    filtered = df[df["user_id"].isin(recent_users)].copy()
    print(f"  temporal subset: kept {len(filtered)}/{len(df)} users (last {fraction*100:.0f}%)")
    return filtered

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-users", type=Path, default=Path("data/preprocessed/train/train_users.csv"))
    p.add_argument("--props", type=Path, default=Path("data/preprocessed/train/train_users_properties.csv"))
    p.add_argument("--countries", type=Path, default=Path("data/countries.csv"))
    p.add_argument("--purchases", type=Path, default=Path("data/purchases_train.csv"))
    p.add_argument("--subscriptions", type=Path, default=Path("data/subscriptions.csv"))
    p.add_argument("--durations", type=Path, default=Path("data/durations_train.csv"))
    p.add_argument("--output", type=Path, default=Path("output/churn_country_correlations.png"))
    p.add_argument("--temporal-fraction", type=float, default=0.3)
    args = p.parse_args()

    print("Load users...")
    users = read_users(args.train_users)
    print(f"  n users: {len(users)}")
    print(f"  churn rate: {users['churn'].mean():.4f}")
    churn_only = users[users["churn"] == 1]
    print(f"  churned users: {len(churn_only)}")
    print(f"  vol_churn share among churned: {churn_only['vol_vs_invol'].mean():.4f}")

    print("Build country features...")
    country_df, b_feat_names, labels_b = build_country_features(args.props, args.countries)
    print(f"  users with country data: {len(country_df)}")

    print("Build purchase features...")
    purchase_df, c_feat_names, labels_c = build_purchase_features(args.purchases)
    print(f"  users with purchase data: {len(purchase_df)}")

    print("Build subscription features...")
    subscription_df, d_feat_names, labels_d = build_subscription_features(args.subscriptions)
    print(f"  users with subscription data: {len(subscription_df)}")

    print("Build duration features...")
    duration_df, e_feat_names, labels_e = build_duration_features(args.durations)
    print(f"  users with duration data: {len(duration_df)}")

    df = users.merge(country_df, on="user_id", how="inner")
    df = df.merge(purchase_df, on="user_id", how="left")
    df = df.merge(subscription_df, on="user_id", how="left")
    df = df.merge(duration_df, on="user_id", how="left")

    print("Temporal subsetting...")
    df = temporal_subset_last_fraction(df, args.props, fraction=args.temporal_fraction)

    print("Build interaction features...")
    i_feat_names, labels_i = build_interaction_features(
        df, b_feat_names, c_feat_names, d_feat_names, e_feat_names,
        labels_b, labels_c, labels_d, labels_e,
    )

    df_churn_only = df[df["churn"] == 1].copy()

    all_feat_names = b_feat_names + c_feat_names + d_feat_names + e_feat_names + i_feat_names
    all_labels = labels_b + labels_c + labels_d + labels_e + labels_i
    nf = len(all_feat_names)

    dep_labels = ["churn (all)", "vol vs invol"]
    nd = len(dep_labels)

    matrix = [[float("nan")] * nd for _ in range(nf)]

    print(f"\n{'Formula':<65} {'churn':>10} {'vol_vs_invol':>12}")
    print("-" * 90)
    for j, (fn, lb) in enumerate(zip(all_feat_names, all_labels)):
        mask_all = df[fn].notna()
        r_churn = pearson(
            df.loc[mask_all, "churn"].astype(float).tolist(),
            df.loc[mask_all, fn].astype(float).tolist(),
        )
        matrix[j][0] = r_churn

        mask_co = df_churn_only[fn].notna()
        r_vol = pearson(
            df_churn_only.loc[mask_co, "vol_vs_invol"].astype(float).tolist(),
            df_churn_only.loc[mask_co, fn].astype(float).tolist(),
        )
        matrix[j][1] = r_vol

        print(f"{lb:<65} {r_churn:>10.4f} {r_vol:>12.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, max(6, nf * 0.55)))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(nd))
    ax.set_xticklabels(dep_labels, fontsize=10)
    ax.set_yticks(range(nf))
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_xlabel("Dependent variable")
    ax.set_ylabel("Features (B=country, C=purchase, D=subscription, E=duration, I=interaction)")
    ax.set_title("Pearson Correlation")

    nb = len(b_feat_names)
    nc = len(c_feat_names)
    nd_feat = len(d_feat_names)
    ne = len(e_feat_names)
    if nc > 0:
        ax.axhline(y=nb - 0.5, color="gray", linewidth=1, linestyle="--")
    if nd_feat > 0:
        ax.axhline(y=nb + nc - 0.5, color="gray", linewidth=1, linestyle="--")
    if ne > 0:
        ax.axhline(y=nb + nc + nd_feat - 0.5, color="gray", linewidth=1, linestyle="--")
    if len(i_feat_names) > 0:
        ax.axhline(y=nb + nc + nd_feat + ne - 0.5, color="gray", linewidth=1, linestyle="--")

    for i in range(nf):
        for j in range(nd):
            val = matrix[i][j]
            if math.isfinite(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")

    plt.tight_layout()
    plt.savefig(str(args.output), dpi=150)
    print(f"\nSaved heatmap to {args.output}")


if __name__ == "__main__":
    main()