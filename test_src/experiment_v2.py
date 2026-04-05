"""
Churn feature screening: Pearson r + mutual information on the LAST fraction of users
when ordered by subscription / anchor time (e.g. last 30% from Aug–Dec cohort).

Merge: train_users + train_user_market_anchor_features (+ optional country/purchase formulas).
"""

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

try:
    from sklearn.feature_selection import mutual_info_classif
except ImportError:
    mutual_info_classif = None

# --- Optional handcrafted features (country / purchase formulas) -----------------

FORMULAS_B = ["11"]

FORMULAS_C = ["2 * 23"]

INTERACTION_FORMULAS = ["c_0 * b_0"]


def tokenize(formula):
    tokens = []
    i = 0
    s = formula.replace(" ", "")
    while i < len(s):
        if s[i].isdigit() or (s[i] == "." and i + 1 < len(s) and s[i + 1].isdigit()):
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(("NUM", s[i:j]))
            i = j
        elif s[i : i + 4] == "sqrt":
            tokens.append(("FUNC", "sqrt"))
            i += 4
        elif s[i : i + 3] == "exp":
            tokens.append(("FUNC", "exp"))
            i += 3
        elif s[i : i + 3] == "abs":
            tokens.append(("FUNC", "abs"))
            i += 3
        elif s[i : i + 3] == "log":
            tokens.append(("FUNC", "log"))
            i += 3
        elif s[i : i + 3] == "pow":
            tokens.append(("FUNC", "pow"))
            i += 3
        elif s[i] in "+-*/^":
            tokens.append(("OP", s[i]))
            i += 1
        elif s[i] == "(":
            tokens.append(("LPAREN", "("))
            i += 1
        elif s[i] == ")":
            tokens.append(("RPAREN", ")"))
            i += 1
        elif s[i] == ",":
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
        if "." not in tok[1]:
            return ("col", int(tok[1]))
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

    return re.sub(r"(?<![.\d])\b(\d+)\b(?!\.\d)", lambda m: replace_col(m), formula)


def pearson(va, vb):
    a = np.asarray(va, dtype=np.float64)
    b = np.asarray(vb, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def read_users_base(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    c0 = df.columns[0]
    if str(c0).startswith("Unnamed") or str(c0).lower() == "skipped":
        df = df.drop(columns=c0)
    df["churn"] = df["churn_status"].isin(["vol_churn", "invol_churn"]).astype(np.int8)
    df["vol_vs_invol"] = np.where(
        df["churn_status"] == "vol_churn",
        1,
        np.where(df["churn_status"] == "invol_churn", 0, np.nan),
    )
    return df


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


def build_interaction_features(df, b_feat_names, c_feat_names, b_labels, c_labels):
    feat_names = []
    labels = []
    for idx, formula in enumerate(INTERACTION_FORMULAS):
        refs = re.findall(r"[bc]_\d+", formula)
        ref_to_slot: dict[str, int] = {}
        slot_to_col: dict[int, str] = {}
        for ref in refs:
            if ref in ref_to_slot:
                continue
            _kind, fidx = ref[0], int(ref[2:])
            if ref[0] == "b":
                col = b_feat_names[fidx]
            else:
                col = c_feat_names[fidx]
            slot = len(ref_to_slot)
            ref_to_slot[ref] = slot
            slot_to_col[slot] = col

        numeric_formula = formula
        for ref in sorted(ref_to_slot, key=len, reverse=True):
            numeric_formula = numeric_formula.replace(ref, str(ref_to_slot[ref]))

        tree = parse_formula(numeric_formula)

        label_formula = formula
        for ref in sorted(ref_to_slot, key=len, reverse=True):
            _kind, fidx = ref[0], int(ref[2:])
            if ref[0] == "b":
                label_formula = label_formula.replace(ref, f"B({b_labels[fidx]})")
            else:
                label_formula = label_formula.replace(ref, f"C({c_labels[fidx]})")

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


def temporal_subset_last_fraction(df: pd.DataFrame, date_col: str, frac: float) -> pd.DataFrame:
    """Keep the most recent `frac` share of rows (e.g. 0.3 → last 30%) by `date_col` ascending."""
    if date_col not in df.columns:
        raise ValueError(f"Missing {date_col} for chronological sort")
    d = df.dropna(subset=[date_col]).copy()
    d[date_col] = pd.to_datetime(d[date_col], utc=True, errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    n = len(d)
    if n == 0:
        return d
    start_idx = int(math.ceil((1.0 - frac) * n))
    return d.iloc[start_idx:].copy()


EXCLUDE_FROM_X = frozenset(
    {
        "user_id",
        "churn_status",
        "churn",
        "vol_vs_invol",
        "anchor_date",
        "subscription_start_ts_raw",
        "subscription_start_ts_decoded",
    }
)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_FROM_X:
            continue
        if df[c].dtype == object:
            continue
        cols.append(c)
    return cols


def compute_mi(X: np.ndarray, y: np.ndarray, random_state: int = 0) -> np.ndarray:
    if mutual_info_classif is None:
        return np.full(X.shape[1], np.nan)
    return mutual_info_classif(X, y, random_state=random_state)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pearson + MI vs churn on last fraction of train (by anchor_date)."
    )
    p.add_argument("--train-users", type=Path, default=Path("data/train/train_users.csv"))
    p.add_argument(
        "--market-anchor",
        type=Path,
        default=Path("output/train_user_market_anchor_features.csv"),
    )
    p.add_argument("--props", type=Path, default=Path("data/train/train_users_properties.csv"))
    p.add_argument("--countries", type=Path, default=Path("data/countries.csv"))
    p.add_argument(
        "--purchases",
        type=Path,
        default=Path("data/train/train_users_purchases.csv"),
    )
    p.add_argument("--date-col", type=str, default="anchor_date", help="Chronological sort key")
    p.add_argument(
        "--last-frac",
        type=float,
        default=0.3,
        help="Fraction of users to keep (most recent by date-col), e.g. 0.3 = last 30%%",
    )
    p.add_argument(
        "--with-extras",
        action="store_true",
        help="Merge country (B) + purchase (C) + interaction (I) handcrafted features",
    )
    p.add_argument("--output-table", type=Path, default=Path("output/experiment_v2_last30_corr_mi.csv"))
    p.add_argument("--output-heatmap", type=Path, default=Path("output/experiment_v2_last30_corr_heatmap.png"))
    args = p.parse_args()

    print("Load train users...")
    base = read_users_base(args.train_users)
    print(f"  n={len(base)}, churn rate={base['churn'].mean():.4f}")

    print("Load market anchor features...")
    mkt = pd.read_csv(args.market_anchor, low_memory=False)
    mkt["user_id"] = mkt["user_id"].astype(str)
    base["user_id"] = base["user_id"].astype(str)
    df = base.merge(mkt, on="user_id", how="inner", suffixes=("", "_mkt"))
    if "churn_status_mkt" in df.columns:
        df = df.drop(columns=[c for c in df.columns if c.endswith("_mkt") and "churn" in c])
    print(f"  after merge: n={len(df)}")

    b_feat_names: list[str] = []
    c_feat_names: list[str] = []
    labels_b: list[str] = []
    labels_c: list[str] = []
    i_feat_names: list[str] = []
    labels_i: list[str] = []

    if args.with_extras:
        print("Build country / purchase / interaction features...")
        country_df, b_feat_names, labels_b = build_country_features(args.props, args.countries)
        country_df["user_id"] = country_df["user_id"].astype(str)
        df = df.merge(country_df, on="user_id", how="left")
        if args.purchases.exists():
            purchase_df, c_feat_names, labels_c = build_purchase_features(args.purchases)
            purchase_df["user_id"] = purchase_df["user_id"].astype(str)
            df = df.merge(purchase_df, on="user_id", how="left")
        else:
            print(f"  [skip] purchases not found: {args.purchases}")
        if b_feat_names and c_feat_names:
            i_feat_names, labels_i = build_interaction_features(
                df, b_feat_names, c_feat_names, labels_b, labels_c
            )

    print(
        f"Temporal subset: last {args.last_frac:.0%} by {args.date_col} "
        f"(most recent subscriptions in train)"
    )
    sub = temporal_subset_last_fraction(df, args.date_col, args.last_frac)
    print(f"  subset n={len(sub)} (from total merged n={len(df)})")
    print(f"  subset churn rate={sub['churn'].mean():.4f}")
    if len(sub) == 0:
        raise SystemExit("Empty subset — check date_col and merges.")

    feat_names = feature_columns(sub)
    if not feat_names:
        raise SystemExit("No numeric feature columns after exclusions.")

    X = sub[feat_names].replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X_imp = X.fillna(med).values.astype(np.float64)
    y = sub["churn"].values.astype(int)

    print("Compute Pearson r vs churn...")
    r_list = []
    for c in feat_names:
        m = sub[c].replace([np.inf, -np.inf], np.nan).notna()
        r_list.append(
            pearson(sub.loc[m, "churn"].astype(float).tolist(), sub.loc[m, c].astype(float).tolist())
        )

    print("Compute MI vs churn...")
    mi = compute_mi(X_imp, y)

    out_tbl = pd.DataFrame(
        {
            "feature": feat_names,
            "pearson_r_vs_churn": r_list,
            "mutual_info_vs_churn": mi,
            "abs_pearson": np.abs(np.array(r_list, dtype=float)),
            "abs_mi": np.abs(mi),
        }
    )
    out_tbl = out_tbl.sort_values("mutual_info_vs_churn", ascending=False, na_position="last")

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    out_tbl.to_csv(args.output_table, index=False)
    print(f"\nSaved table -> {args.output_table}")
    print("\nTop 15 by MI:")
    print(out_tbl.head(15).to_string(index=False))

    # Heatmap: top 40 features by MI, two columns: churn r, MI (scaled)
    top_k = min(40, len(out_tbl))
    top = out_tbl.head(top_k)
    mat = np.zeros((top_k, 2))
    mat[:, 0] = top["pearson_r_vs_churn"].values
    # MI typically smaller — scale for display
    mi_max = np.nanmax(np.abs(top["mutual_info_vs_churn"].values))
    mat[:, 1] = top["mutual_info_vs_churn"].values / (mi_max if mi_max and np.isfinite(mi_max) else 1.0)

    fig, ax = plt.subplots(figsize=(8, max(6, top_k * 0.35)))
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pearson r (churn)", "MI (norm)"], fontsize=10)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top["feature"].values, fontsize=7)
    ax.set_title(
        f"Last {args.last_frac:.0%} of train by {args.date_col} (n={len(sub)})\n"
        "Top features by MI (rows)"
    )
    for i in range(top_k):
        for j in range(2):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(val) > 0.5 else "black",
                )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(str(args.output_heatmap), dpi=150)
    print(f"Saved heatmap -> {args.output_heatmap}")

    # Vol vs invol among churned (subset) — Pearson only, small table
    ch = sub[sub["churn"] == 1].copy()
    if ch["vol_vs_invol"].notna().sum() > 10:
        print("\nPearson vs vol_vs_invol (churned only, same subset) - top 10 by |r|:")
        rv = []
        for c in feat_names:
            m = ch[c].replace([np.inf, -np.inf], np.nan).notna() & ch["vol_vs_invol"].notna()
            if m.sum() < 10:
                continue
            rv.append(
                (
                    c,
                    pearson(
                        ch.loc[m, "vol_vs_invol"].astype(float).tolist(),
                        ch.loc[m, c].astype(float).tolist(),
                    ),
                )
            )
        rv = sorted(rv, key=lambda x: abs(x[1]) if np.isfinite(x[1]) else -1, reverse=True)[:10]
        for name, rr in rv:
            print(f"  {name[:55]:<55} {rr:>8.4f}")


if __name__ == "__main__":
    main()
