"""Run all experiments on LAST FOLD ONLY (fold 4 of TimeSeriesSplit(5))."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_TS_SPLITS = 5
DATA_DIR = Path("/home/ansar/work/hack-nu-26/data/preprocessed/train")

with open("/home/ansar/work/hack-nu-26/best_model_params.json") as f:
    BEST_PARAMS = json.load(f)

# ── Data loading (same as training_pipeline) ──
print("Loading data...")
u = pd.read_csv(DATA_DIR / "train_users.csv",
    usecols=["user_id", "churn_status", "gdp_growth_pct", "log_churn_density"])
u["gdp_growth_pct"] = pd.to_numeric(u["gdp_growth_pct"], errors="coerce")
u["log_churn_density"] = pd.to_numeric(u["log_churn_density"], errors="coerce")

_gen = pd.read_csv(DATA_DIR.parent.parent / "train" / "train_users.csv",
    usecols=["user_id", "gen_delta_day1_minus_day14", "log1p_total_gen"],
    low_memory=False).drop_duplicates(subset=["user_id"], keep="first")
u = u.merge(_gen, on="user_id", how="left")
u["gen_delta_day1_minus_day14"] = pd.to_numeric(u["gen_delta_day1_minus_day14"], errors="coerce")
u["log1p_total_gen"] = pd.to_numeric(u["log1p_total_gen"], errors="coerce")

prop = pd.read_csv(DATA_DIR / "train_users_properties.csv",
    usecols=["user_id", "subscription_start_date", "subscription_plan", "country_code"])
q = pd.read_csv(DATA_DIR / "train_users_quizzes.csv",
    usecols=["user_id", "source", "team_size", "experience",
             "usage_plan", "frustration", "first_feature", "role"])
df = u.merge(prop, on="user_id", how="left").merge(q, on="user_id", how="left")
df["subscription_start_ts"] = pd.to_datetime(
    df["subscription_start_date"], utc=True, errors="coerce").astype("int64")
df = df.drop(columns=["subscription_start_date"])

pur = pd.read_csv(DATA_DIR / "train_users_purchases.csv",
    usecols=["user_id", "transaction_id", "purchase_type", "purchase_amount_dollars"],
    low_memory=False)
ta = pd.read_csv(DATA_DIR / "train_users_transaction_attempts.csv",
    usecols=["transaction_id", "amount_in_usd", "billing_address_country",
             "card_3d_secure_support", "card_country", "card_funding", "cvc_check",
             "digital_wallet", "is_3d_secure", "is_3d_secure_authenticated",
             "payment_method_type", "bank_country", "is_prepaid", "is_virtual", "is_business"],
    low_memory=False)
for c in ta.columns:
    if c.startswith("is_"):
        ta[c] = ta[c].map(lambda x: str(x).lower() in {"true", "1"})
m = pur.merge(ta, on="transaction_id", how="left", suffixes=("_pur", ""))
bool_cols = [c for c in ta.columns if c.startswith("is_")]
m["att_card_flags_row"] = m[bool_cols].astype(np.float64).mean(axis=1)
mix_cols = [c for c in ["billing_address_country", "card_country", "card_funding",
    "payment_method_type", "cvc_check", "digital_wallet",
    "card_3d_secure_support", "bank_country"] if c in m.columns]
m["att_payment_mix_key"] = m[mix_cols].astype(str).apply(lambda r: "|".join(r.values), axis=1)
g = m.groupby("user_id", as_index=False).agg({
    "transaction_id": "count", "purchase_amount_dollars": "sum",
    "purchase_type": "nunique", "amount_in_usd": "sum",
    "att_card_flags_row": "mean", "att_payment_mix_key": "nunique",
}).rename(columns={
    "transaction_id": "purch_n", "purchase_amount_dollars": "purch_amount_sum",
    "purchase_type": "purch_type_nunique", "amount_in_usd": "att_amount_sum",
    "att_card_flags_row": "att_card_flags_mean", "att_payment_mix_key": "att_payment_mix_nunique",
})
df = df.merge(g, on="user_id", how="left")

path = DATA_DIR / "train_users_generations.csv"
sp, tp = [], []
for chunk in pd.read_csv(path, chunksize=2_000_000, usecols=["user_id", "status", "generation_type"]):
    sp.append(chunk.groupby(["user_id", "status"]).size())
    tp.append(chunk.groupby(["user_id", "generation_type"]).size())
st = pd.concat(sp).groupby(level=[0, 1]).sum().unstack(fill_value=0)
st.columns = [f"gen_status_{c}" for c in st.columns.astype(str)]
gt = pd.concat(tp).groupby(level=[0, 1]).sum().unstack(fill_value=0)
gt.columns = [f"gen_type_{c}" for c in gt.columns.astype(str)]
out = st.join(gt, how="outer").fillna(0).astype(np.float32)
out["gen_total"] = out[[c for c in out.columns if c.startswith("gen_status_")]].sum(axis=1)
df = df.merge(out.reset_index(), on="user_id", how="left")

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols and c not in {"user_id", "churn_status"}]
for c in cat_cols:
    df[c] = df[c].fillna("skipped").astype(str)
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df = df.sort_values("subscription_start_ts").reset_index(drop=True)
print(f"Data: {df.shape}")

# ── Setup ──
y_stay = (df["churn_status"] == "not_churned").astype(np.int8).values
y_vol = (df["churn_status"] == "vol_churn").astype(np.int8).values
feature_cols = [c for c in df.columns if c not in {"user_id", "churn_status"}]
X_df = df[feature_cols].copy()
num_features = [c for c in feature_cols if c in num_cols]
cat_features = [c for c in feature_cols if c in cat_cols]
for c in cat_features:
    X_df[c] = X_df[c].fillna("skipped").astype(str)

STAGE1_DROP = frozenset({"log_churn_density", "log1p_total_gen"})
STAGE2_DROP = frozenset({"gen_delta_day1_minus_day14"})
cols_stage1 = [c for c in feature_cols if c not in STAGE1_DROP]
cols_stage2 = [c for c in feature_cols if c not in STAGE2_DROP]
num_stage1 = [c for c in cols_stage1 if c in num_cols]
cat_stage1 = [c for c in cols_stage1 if c in cat_cols]
num_stage2 = [c for c in cols_stage2 if c in num_cols]
cat_stage2 = [c for c in cols_stage2 if c in cat_cols]

# Get LAST fold indices
tss = TimeSeriesSplit(n_splits=N_TS_SPLITS)
folds = list(tss.split(X_df))
tr, va = folds[-1]  # last fold only
print(f"Last fold: train={len(tr)}, val={len(va)}")


def make_preprocessor(scale, *, num_cols_arg=None, cat_cols_arg=None):
    nf = num_cols_arg or num_features
    cf = cat_cols_arg or cat_features
    return ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]
                         + ([("sc", StandardScaler())] if scale else [])), nf),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cf),
    ], remainder="drop", sparse_threshold=0.0)


def hierarchical_class_proba(p_stay, p_vol_if_churn):
    p_stay = np.clip(p_stay, 1e-8, 1 - 1e-8)
    p_vol_if_churn = np.clip(p_vol_if_churn, 1e-8, 1 - 1e-8)
    pc = 1.0 - p_stay
    return np.column_stack([pc * (1 - p_vol_if_churn), pc * p_vol_if_churn, p_stay])


def _cls_metrics(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    pred = (p >= 0.5).astype(np.int8)
    return (100.0 * accuracy_score(y, pred),
            100.0 * f1_score(y, pred, pos_label=1, zero_division=0),
            100.0 * roc_auc_score(y, p) if np.unique(y).size >= 2 else float("nan"))


def fit_eval(X_tr, X_va, ys_tr, ys_va, yv_tr, yv_va, churned_tr, mode,
             cs1, cs2, ns1, cas1, ns2, cas2, weights=None):
    bp = BEST_PARAMS.get(mode, {})
    bp1, bp2 = bp.get("stage1", {}), bp.get("stage2", {})
    res = {}

    if mode == "catboost":
        X1 = X_tr[cs1].copy(); Xv1 = X_va[cs1].copy()
        for c in cas1:
            X1[c] = X1[c].fillna("skipped").astype(str).astype(object)
            Xv1[c] = Xv1[c].fillna("skipped").astype(str).astype(object)
        ci1 = [X1.columns.get_loc(c) for c in cas1]
        m1 = CatBoostClassifier(loss_function="Logloss",
            iterations=int(bp1.get("iterations", 548)), depth=int(bp1.get("depth", 7)),
            learning_rate=float(bp1.get("learning_rate", 0.028)),
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
        pkw = dict(cat_features=ci1)
        if weights is not None: pkw["weight"] = weights
        m1.fit(Pool(X1, label=ys_tr, **pkw))
        ps = m1.predict_proba(Xv1)[:, 1]

        X2 = X_tr.iloc[churned_tr][cs2].copy(); Xv2 = X_va[cs2].copy()
        for c in cas2:
            X2[c] = X2[c].fillna("skipped").astype(str).astype(object)
            Xv2[c] = Xv2[c].fillna("skipped").astype(str).astype(object)
        ci2 = [X2.columns.get_loc(c) for c in cas2]
        pkw2 = dict(cat_features=ci2)
        if weights is not None: pkw2["weight"] = weights[churned_tr]
        m2 = CatBoostClassifier(loss_function="Logloss",
            iterations=int(bp2.get("iterations", 520)), depth=int(bp2.get("depth", 8)),
            learning_rate=float(bp2.get("learning_rate", 0.060)),
            random_seed=RANDOM_STATE+1, verbose=False, allow_writing_files=False)
        m2.fit(Pool(X2, label=yv_tr[churned_tr], **pkw2))
        pv = m2.predict_proba(Xv2)[:, 1]

    elif mode == "xgb":
        pre1 = make_preprocessor(False, num_cols_arg=ns1, cat_cols_arg=cas1)
        Xt1 = pre1.fit_transform(X_tr[cs1]); Xv1 = pre1.transform(X_va[cs1])
        m1 = XGBClassifier(n_estimators=int(bp1.get("n_estimators", 330)),
            max_depth=int(bp1.get("max_depth", 4)),
            learning_rate=float(bp1.get("learning_rate", 0.050)),
            subsample=float(bp1.get("subsample", 0.913)),
            colsample_bytree=float(bp1.get("colsample_bytree", 0.718)),
            reg_lambda=float(bp1.get("reg_lambda", 3.185)),
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0)
        fkw = {}
        if weights is not None: fkw["sample_weight"] = weights
        m1.fit(Xt1, ys_tr, **fkw)
        ps = m1.predict_proba(Xv1)[:, 1]

        pre2 = make_preprocessor(False, num_cols_arg=ns2, cat_cols_arg=cas2)
        pre2.fit(X_tr[cs2])
        Xt2 = pre2.transform(X_tr[cs2])[churned_tr]; Xv2 = pre2.transform(X_va[cs2])
        fkw2 = {}
        if weights is not None: fkw2["sample_weight"] = weights[churned_tr]
        m2 = XGBClassifier(n_estimators=int(bp2.get("n_estimators", 394)),
            max_depth=int(bp2.get("max_depth", 7)),
            learning_rate=float(bp2.get("learning_rate", 0.020)),
            subsample=float(bp2.get("subsample", 0.883)),
            colsample_bytree=float(bp2.get("colsample_bytree", 0.700)),
            reg_lambda=float(bp2.get("reg_lambda", 4.322)),
            random_state=RANDOM_STATE+1, n_jobs=-1, eval_metric="logloss", verbosity=0)
        m2.fit(Xt2, yv_tr[churned_tr], **fkw2)
        pv = m2.predict_proba(Xv2)[:, 1]

    p3 = hierarchical_class_proba(ps, pv)
    yc = (ys_va == 0).astype(np.int8)
    pc = p3[:, 0] + p3[:, 1]
    a, f, r = _cls_metrics(yc, pc)
    res["acc"] = a; res["f1"] = f; res["auc"] = r
    mc = ys_va == 0
    if mc.sum() > 0:
        s2a, s2f, s2r = _cls_metrics(yv_va[mc], pv[mc])
        res["s2_acc"] = s2a; res["s2_f1"] = s2f; res["s2_auc"] = s2r
    return res, p3


def run_single_fold(label, X_use=None, cs1_=None, cs2_=None, ns1_=None, cas1_=None,
                    ns2_=None, cas2_=None, weight_fn=None, tr_filter=None):
    X = X_use if X_use is not None else X_df
    c1 = cs1_ or cols_stage1; c2 = cs2_ or cols_stage2
    n1 = ns1_ or num_stage1; ca1 = cas1_ or cat_stage1
    n2 = ns2_ or num_stage2; ca2 = cas2_ or cat_stage2
    tr_use = tr_filter(tr) if tr_filter else tr
    sw = weight_fn(tr_use) if weight_fn else None
    Xt, Xv = X.iloc[tr_use], X.iloc[va]
    yst, ysv = y_stay[tr_use], y_stay[va]
    yvt, yvv = y_vol[tr_use], y_vol[va]
    ct = (df["churn_status"].iloc[tr_use] != "not_churned").values
    rows = []
    for mode in ("catboost", "xgb"):
        r, _ = fit_eval(Xt, Xv, yst, ysv, yvt, yvv, ct, mode, c1, c2, n1, ca1, n2, ca2, sw)
        r["model"] = mode
        r["experiment"] = label
        rows.append(r)
    return rows


# ── Run all experiments on LAST FOLD ──
print("\n" + "=" * 70)
print("  LAST-FOLD-ONLY EXPERIMENTS (train on 80%, val on latest 20%)")
print("=" * 70)

all_rows = []

print("\n>> 0_baseline")
all_rows += run_single_fold("0_baseline")

print(">> 1_sample_weight_linear")
all_rows += run_single_fold("1_sample_weight_linear",
    weight_fn=lambda idx: np.linspace(0.5, 1.5, len(idx)))

print(">> 1b_sample_weight_exp")
def _wexp(idx):
    raw = np.exp(np.linspace(0, 1.5, len(idx)))
    return raw / raw.mean()
all_rows += run_single_fold("1b_sample_weight_exp", weight_fn=_wexp)

print(">> 2_drop_country")
drop = {"country_code"}
all_rows += run_single_fold("2_drop_country",
    cs1_=[c for c in cols_stage1 if c not in drop],
    cs2_=[c for c in cols_stage2 if c not in drop],
    cas1_=[c for c in cat_stage1 if c not in drop],
    cas2_=[c for c in cat_stage2 if c not in drop])

print(">> 3_interactions")
X3 = X_df.copy()
X3["avg_purchase"] = X3["purch_amount_sum"] / X3["purch_n"].clip(lower=1)
X3["flags_x_purch"] = X3["att_card_flags_mean"] * X3["purch_n"]
X3["log_purch_amount"] = np.log1p(X3["purch_amount_sum"].fillna(0).clip(lower=0))
X3["frustration_x_plan"] = (X3["frustration"].fillna("skipped").astype(str) + "_"
                            + X3["usage_plan"].fillna("skipped").astype(str))
nn = ["avg_purchase", "flags_x_purch", "log_purch_amount"]
nc = ["frustration_x_plan"]
all_rows += run_single_fold("3_interactions", X_use=X3,
    cs1_=cols_stage1 + nn + nc, cs2_=cols_stage2 + nn + nc,
    ns1_=num_stage1 + nn, cas1_=cat_stage1 + nc,
    ns2_=num_stage2 + nn, cas2_=cat_stage2 + nc)

print(">> 4_late_only_50pct")
all_rows += run_single_fold("4_late_only_50pct",
    tr_filter=lambda idx: idx[len(idx) // 2:])

print(">> 5_temporal_ensemble")
ALPHA = 0.4
tr_late = tr[len(tr) // 2:]
Xt_f, Xt_l, Xv_ = X_df.iloc[tr], X_df.iloc[tr_late], X_df.iloc[va]
for mode in ("catboost", "xgb"):
    _, p3f = fit_eval(Xt_f, Xv_, y_stay[tr], y_stay[va], y_vol[tr], y_vol[va],
        (df["churn_status"].iloc[tr] != "not_churned").values, mode,
        cols_stage1, cols_stage2, num_stage1, cat_stage1, num_stage2, cat_stage2)
    _, p3l = fit_eval(Xt_l, Xv_, y_stay[tr_late], y_stay[va], y_vol[tr_late], y_vol[va],
        (df["churn_status"].iloc[tr_late] != "not_churned").values, mode,
        cols_stage1, cols_stage2, num_stage1, cat_stage1, num_stage2, cat_stage2)
    p3b = ALPHA * p3f + (1 - ALPHA) * p3l
    yc = (y_stay[va] == 0).astype(np.int8)
    pc = p3b[:, 0] + p3b[:, 1]
    a, f, r = _cls_metrics(yc, pc)
    row = {"acc": a, "f1": f, "auc": r, "model": mode, "experiment": "5_temporal_ensemble"}
    mc = y_stay[va] == 0
    if mc.sum() > 0:
        pv_b = p3b[mc, 1] / (p3b[mc, 0] + p3b[mc, 1]).clip(1e-8)
        s2a, s2f, s2r = _cls_metrics(y_vol[va][mc], pv_b)
        row["s2_acc"] = s2a; row["s2_f1"] = s2f; row["s2_auc"] = s2r
    all_rows.append(row)

print(">> 6_target_encode_country")
SMOOTH = 20
te_drop = {"country_code"}
te_cs1 = [c for c in cols_stage1 if c not in te_drop] + ["country_churn_rate"]
te_cs2 = [c for c in cols_stage2 if c not in te_drop] + ["country_churn_rate"]
te_ns1 = num_stage1 + ["country_churn_rate"]
te_cas1 = [c for c in cat_stage1 if c not in te_drop]
te_ns2 = num_stage2 + ["country_churn_rate"]
te_cas2 = [c for c in cat_stage2 if c not in te_drop]
Xte_tr = X_df.iloc[tr].copy(); Xte_va = X_df.iloc[va].copy()
y_ch = (y_stay[tr] == 0).astype(np.float64)
gp = y_ch.mean()
cc = Xte_tr["country_code"].values
te_map = {}
for cv in np.unique(cc):
    mask = cc == cv; cnt = mask.sum()
    te_map[cv] = (cnt * y_ch[mask].mean() + SMOOTH * gp) / (cnt + SMOOTH)
Xte_tr["country_churn_rate"] = Xte_tr["country_code"].map(te_map).fillna(gp).astype(np.float64)
Xte_va["country_churn_rate"] = Xte_va["country_code"].map(te_map).fillna(gp).astype(np.float64)
for mode in ("catboost", "xgb"):
    r, _ = fit_eval(Xte_tr, Xte_va, y_stay[tr], y_stay[va], y_vol[tr], y_vol[va],
        (df["churn_status"].iloc[tr] != "not_churned").values, mode,
        te_cs1, te_cs2, te_ns1, te_cas1, te_ns2, te_cas2)
    r["model"] = mode; r["experiment"] = "6_target_encode_country"
    all_rows.append(r)

print(">> 7_combo (interactions + drop country + linear weights)")
X7 = X_df.copy()
X7["avg_purchase"] = X7["purch_amount_sum"] / X7["purch_n"].clip(lower=1)
X7["flags_x_purch"] = X7["att_card_flags_mean"] * X7["purch_n"]
X7["log_purch_amount"] = np.log1p(X7["purch_amount_sum"].fillna(0).clip(lower=0))
X7["frustration_x_plan"] = (X7["frustration"].fillna("skipped").astype(str) + "_"
                            + X7["usage_plan"].fillna("skipped").astype(str))
dc = {"country_code"}
nn7 = ["avg_purchase", "flags_x_purch", "log_purch_amount"]
nc7 = ["frustration_x_plan"]
all_rows += run_single_fold("7_combo",
    X_use=X7,
    cs1_=[c for c in cols_stage1 if c not in dc] + nn7 + nc7,
    cs2_=[c for c in cols_stage2 if c not in dc] + nn7 + nc7,
    ns1_=[c for c in num_stage1 if c not in dc] + nn7,
    cas1_=[c for c in cat_stage1 if c not in dc] + nc7,
    ns2_=[c for c in num_stage2 if c not in dc] + nn7,
    cas2_=[c for c in cat_stage2 if c not in dc] + nc7,
    weight_fn=lambda idx: np.linspace(0.5, 1.5, len(idx)))

# ── Results ──
res = pd.DataFrame(all_rows)
print("\n" + "=" * 70)
print("  LAST FOLD RESULTS (train ~74k, val ~15k most recent users)")
print("=" * 70)

for mode in ("catboost", "xgb"):
    sub = res[res["model"] == mode][["experiment", "acc", "f1", "auc", "s2_f1", "s2_auc"]].copy()
    sub = sub.sort_values("auc", ascending=False).reset_index(drop=True)
    print(f"\n  [{mode}] ranked by AUC:")
    base_auc = sub.loc[sub["experiment"] == "0_baseline", "auc"].values[0]
    sub["delta_auc"] = sub["auc"] - base_auc
    for _, r in sub.iterrows():
        d = r["delta_auc"]
        sign = "+" if d >= 0 else ""
        print(f"    {r['experiment']:30s}  acc={r['acc']:.2f}  f1={r['f1']:.2f}  "
              f"auc={r['auc']:.2f} ({sign}{d:.2f})  |  s2_f1={r.get('s2_f1', float('nan')):.2f}  "
              f"s2_auc={r.get('s2_auc', float('nan')):.2f}")
