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


FORMULAS_B = [
    "exp(14 / 8)",
    "exp(15 / 8)", 
    "14 / 8",
    "15 / 8", 
    "log(14/8)",
    "log(15/8)",
    "(14 + 15)",
    "(14 + 15) / 8",
    "log((14 + 15) / 8)",
    "14 / 8 * 11",
    "14 / 8 / 11",
    "11 / 14 * 8",
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
    n = len(va)
    if n < 2:
        return float("nan")
    ma = sum(va) / n
    mb = sum(vb) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(va, vb)) / (n - 1)
    sa = math.sqrt(sum((a - ma) ** 2 for a in va) / (n - 1))
    sb = math.sqrt(sum((b - mb) ** 2 for b in vb) / (n - 1))
    if sa == 0 or sb == 0:
        return float("nan")
    return cov / (sa * sb)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-users", type=Path, default=Path("data/preprocessed/train/train_users.csv"))
    p.add_argument("--props", type=Path, default=Path("data/preprocessed/train/train_users_properties.csv"))
    p.add_argument("--countries", type=Path, default=Path("data/countries.csv"))
    p.add_argument("--output", type=Path, default=Path("output/churn_country_correlations.png"))
    args = p.parse_args()

    print("Load users...")
    users = read_users(args.train_users)
    print(f"  n users: {len(users)}")
    print(f"  churn rate: {users['churn'].mean():.4f}")
    churn_only = users[users["churn"] == 1]
    print(f"  churned users: {len(churn_only)}")
    print(f"  vol_churn share among churned: {churn_only['vol_vs_invol'].mean():.4f}")

    print("Build country features...")
    country_df, feat_names, labels_b = build_country_features(args.props, args.countries)
    print(f"  users with country data: {len(country_df)}")

    df = users.merge(country_df, on="user_id", how="inner")
    df_churn_only = df[df["churn"] == 1].copy()

    nb = len(feat_names)
    dep_labels = ["churn (all)", "vol vs invol"]
    nd = len(dep_labels)

    matrix = [[float("nan")] * nd for _ in range(nb)]

    print(f"\n{'Formula B':<55} {'churn':>10} {'vol_vs_invol':>12}")
    print("-" * 80)
    for j, (fn, lb) in enumerate(zip(feat_names, labels_b)):
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

        print(f"{lb:<55} {r_churn:>10.4f} {r_vol:>12.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, max(6, nb * 0.55)))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(nd))
    ax.set_xticklabels(dep_labels, fontsize=10)
    ax.set_yticks(range(nb))
    ax.set_yticklabels(labels_b, fontsize=8)
    ax.set_xlabel("Dependent variable")
    ax.set_ylabel("Country formula B features")
    ax.set_title("Pearson Correlation")

    for i in range(nb):
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