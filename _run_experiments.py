# === CELL 0 ===
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_TS_SPLITS = 5
DATA_DIR = Path("/home/ansar/work/hack-nu-26/data/preprocessed/train")
BEST_PARAMS_PATH = Path("/home/ansar/work/hack-nu-26/best_model_params.json")

with open(BEST_PARAMS_PATH) as f:
    BEST_PARAMS = json.load(f)

print("Loaded tuned hyperparams from", BEST_PARAMS_PATH.name)
for k in ("catboost", "xgb"):
    for stage in ("stage1", "stage2"):
        print(f"  {k}/{stage}: {BEST_PARAMS[k][stage]}")

all_experiment_results = []


# === CELL 1 ===
def load_base_users() -> pd.DataFrame:
    u = pd.read_csv(
        DATA_DIR / "train_users.csv",
        usecols=["user_id", "churn_status", "gdp_growth_pct", "log_churn_density"],
    )
    u["gdp_growth_pct"] = pd.to_numeric(u["gdp_growth_pct"], errors="coerce")
    u["log_churn_density"] = pd.to_numeric(u["log_churn_density"], errors="coerce")
    _gen_path = DATA_DIR.parent.parent / "train" / "train_users.csv"
    _gen = pd.read_csv(
        _gen_path,
        usecols=["user_id", "gen_delta_day1_minus_day14", "log1p_total_gen"],
        low_memory=False,
    ).drop_duplicates(subset=["user_id"], keep="first")
    u = u.merge(_gen, on="user_id", how="left")
    u["gen_delta_day1_minus_day14"] = pd.to_numeric(u["gen_delta_day1_minus_day14"], errors="coerce")
    u["log1p_total_gen"] = pd.to_numeric(u["log1p_total_gen"], errors="coerce")
    prop = pd.read_csv(
        DATA_DIR / "train_users_properties.csv",
        usecols=["user_id", "subscription_start_date", "subscription_plan", "country_code"],
    )
    q = pd.read_csv(
        DATA_DIR / "train_users_quizzes.csv",
        usecols=[
            "user_id", "source", "team_size", "experience",
            "usage_plan", "frustration", "first_feature", "role",
        ],
    )
    df = u.merge(prop, on="user_id", how="left").merge(q, on="user_id", how="left")
    df["subscription_start_ts"] = pd.to_datetime(
        df["subscription_start_date"], utc=True, errors="coerce"
    ).astype("int64")
    return df.drop(columns=["subscription_start_date"])


def aggregate_purchases_attempts() -> pd.DataFrame:
    pur = pd.read_csv(
        DATA_DIR / "train_users_purchases.csv",
        usecols=["user_id", "transaction_id", "purchase_type", "purchase_amount_dollars"],
        low_memory=False,
    )
    ta_use = [
        "transaction_id", "amount_in_usd", "billing_address_country",
        "card_3d_secure_support", "card_country", "card_funding", "cvc_check",
        "digital_wallet", "is_3d_secure", "is_3d_secure_authenticated",
        "payment_method_type", "bank_country", "is_prepaid", "is_virtual", "is_business",
    ]
    ta = pd.read_csv(
        DATA_DIR / "train_users_transaction_attempts.csv", usecols=ta_use, low_memory=False,
    )
    for c in ta.columns:
        if c.startswith("is_"):
            ta[c] = ta[c].map(lambda x: str(x).lower() in {"true", "1"})
    m = pur.merge(ta, on="transaction_id", how="left", suffixes=("_pur", ""))
    bool_cols = [c for c in ta.columns if c.startswith("is_")]
    m["att_card_flags_row"] = m[bool_cols].astype(np.float64).mean(axis=1) if bool_cols else 0.0
    mix_cols = [
        c for c in [
            "billing_address_country", "card_country", "card_funding",
            "payment_method_type", "cvc_check", "digital_wallet",
            "card_3d_secure_support", "bank_country",
        ] if c in m.columns
    ]
    m["att_payment_mix_key"] = (
        m[mix_cols].astype(str).apply(lambda r: "|".join(r.values), axis=1)
        if mix_cols else ""
    )
    agg_dict = {
        "transaction_id": "count", "purchase_amount_dollars": "sum",
        "purchase_type": "nunique", "amount_in_usd": "sum",
        "att_card_flags_row": "mean", "att_payment_mix_key": "nunique",
    }
    g = m.groupby("user_id", as_index=False).agg(agg_dict)
    return g.rename(columns={
        "transaction_id": "purch_n", "purchase_amount_dollars": "purch_amount_sum",
        "purchase_type": "purch_type_nunique", "amount_in_usd": "att_amount_sum",
        "att_card_flags_row": "att_card_flags_mean",
        "att_payment_mix_key": "att_payment_mix_nunique",
    })


def aggregate_generations(chunksize=2_000_000) -> pd.DataFrame:
    path = DATA_DIR / "train_users_generations.csv"
    usecols = ["user_id", "status", "generation_type"]
    status_parts, type_parts = [], []
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        status_parts.append(chunk.groupby(["user_id", "status"]).size())
        type_parts.append(chunk.groupby(["user_id", "generation_type"]).size())
    st = pd.concat(status_parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    st.columns = [f"gen_status_{c}" for c in st.columns.astype(str)]
    gt = pd.concat(type_parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    gt.columns = [f"gen_type_{c}" for c in gt.columns.astype(str)]
    out = st.join(gt, how="outer").fillna(0).astype(np.float32)
    out["gen_total"] = out[[c for c in out.columns if c.startswith("gen_status_")]].sum(axis=1)
    return out.reset_index()


df = load_base_users()
pa = aggregate_purchases_attempts()
df = df.merge(pa, on="user_id", how="left")
gen = aggregate_generations()
df = df.merge(gen, on="user_id", how="left")

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in num_cols and c not in {"user_id", "churn_status"}]
for c in cat_cols:
    df[c] = df[c].fillna("skipped").astype(str)
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df = df.sort_values("subscription_start_ts").reset_index(drop=True)

print(df.shape, "numeric:", len(num_cols), "categorical:", len(cat_cols))


# === CELL 2 ===
y_binary_stay = (df["churn_status"] == "not_churned").astype(np.int8).values
y_churn_subtype = (df["churn_status"] == "vol_churn").astype(np.int8).values

feature_cols = [c for c in df.columns if c not in {"user_id", "churn_status"}]
X_df = df[feature_cols].copy()

num_features = [c for c in feature_cols if c in num_cols]
cat_features = [c for c in feature_cols if c in cat_cols]

STAGE1_DROP = frozenset({"log_churn_density", "log1p_total_gen"})
STAGE2_DROP = frozenset({"gen_delta_day1_minus_day14"})
cols_stage1 = [c for c in feature_cols if c not in STAGE1_DROP]
cols_stage2 = [c for c in feature_cols if c not in STAGE2_DROP]
num_stage1 = [c for c in cols_stage1 if c in num_cols]
cat_stage1 = [c for c in cols_stage1 if c in cat_cols]
num_stage2 = [c for c in cols_stage2 if c in num_cols]
cat_stage2 = [c for c in cols_stage2 if c in cat_cols]

for c in cat_features:
    X_df[c] = X_df[c].fillna("skipped").astype(str)


def make_preprocessor(scale, *, num_cols_arg=None, cat_cols_arg=None):
    nf = num_features if num_cols_arg is None else num_cols_arg
    cf = cat_features if cat_cols_arg is None else cat_cols_arg
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]
                             + ([("sc", StandardScaler())] if scale else [])), nf),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cf),
        ],
        remainder="drop", sparse_threshold=0.0,
    )


def hierarchical_class_proba(p_stay, p_vol_if_churn):
    p_stay = np.clip(p_stay, 1e-8, 1 - 1e-8)
    p_vol_if_churn = np.clip(p_vol_if_churn, 1e-8, 1 - 1e-8)
    p_churn = 1.0 - p_stay
    return np.column_stack([p_churn * (1 - p_vol_if_churn), p_churn * p_vol_if_churn, p_stay])


def metrics_binary_churn_pct(y_stay_val, proba3):
    y_churn = (y_stay_val == 0).astype(np.int8)
    p_churn = proba3[:, 0] + proba3[:, 1]
    pred = (p_churn >= 0.5).astype(np.int8)
    return {
        "accuracy_pct": 100.0 * accuracy_score(y_churn, pred),
        "f1_churn_pct": 100.0 * f1_score(y_churn, pred, pos_label=1, zero_division=0),
        "roc_auc_churn_pct": 100.0 * roc_auc_score(y_churn, p_churn),
    }


def _cls_metrics(y, p):
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    pred = (p >= 0.5).astype(np.int8)
    acc = 100.0 * accuracy_score(y, pred)
    f1v = 100.0 * f1_score(y, pred, pos_label=1, zero_division=0)
    auc = 100.0 * roc_auc_score(y, p) if np.unique(y).size >= 2 else float("nan")
    return acc, f1v, auc

print("Features:", len(feature_cols), "| Num:", len(num_features), "| Cat:", len(cat_features))
print("Cat features:", cat_features)


# === CELL 3 ===
def _fit_predict_twostage(
    X_train, X_val, ys_tr, ys_va, yv_tr, yv_va,
    churned_tr, mode, cs1, cs2, ns1, cas1, ns2, cas2,
    weights=None,
):
    diag = {}

    bp = BEST_PARAMS.get(mode, {})
    bp_s1 = bp.get("stage1", {})
    bp_s2 = bp.get("stage2", {})

    if mode == "catboost":
        X1 = X_train[cs1].copy()
        Xv1 = X_val[cs1].copy()
        for c in cas1:
            X1[c] = X1[c].fillna("skipped").astype(str).astype(object)
            Xv1[c] = Xv1[c].fillna("skipped").astype(str).astype(object)
        cat_idx1 = [X1.columns.get_loc(c) for c in cas1]

        m1 = CatBoostClassifier(
            loss_function="Logloss",
            iterations=int(bp_s1.get("iterations", 548)),
            depth=int(bp_s1.get("depth", 7)),
            learning_rate=float(bp_s1.get("learning_rate", 0.028)),
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
        )
        pool_kw1 = dict(cat_features=cat_idx1)
        if weights is not None:
            pool_kw1["weight"] = weights
        m1.fit(Pool(X1, label=ys_tr, **pool_kw1))
        p_stay_val = m1.predict_proba(Xv1)[:, 1]

        X2 = X_train.iloc[churned_tr][cs2].copy()
        Xv2 = X_val[cs2].copy()
        for c in cas2:
            X2[c] = X2[c].fillna("skipped").astype(str).astype(object)
            Xv2[c] = Xv2[c].fillna("skipped").astype(str).astype(object)
        cat_idx2 = [X2.columns.get_loc(c) for c in cas2]
        y_tr2 = yv_tr[churned_tr]
        pool_kw2 = dict(cat_features=cat_idx2)
        if weights is not None:
            pool_kw2["weight"] = weights[churned_tr]
        m2 = CatBoostClassifier(
            loss_function="Logloss",
            iterations=int(bp_s2.get("iterations", 520)),
            depth=int(bp_s2.get("depth", 8)),
            learning_rate=float(bp_s2.get("learning_rate", 0.060)),
            random_seed=RANDOM_STATE + 1, verbose=False, allow_writing_files=False,
        )
        m2.fit(Pool(X2, label=y_tr2, **pool_kw2))
        p_vol_if_churn_val = m2.predict_proba(Xv2)[:, 1]

    elif mode == "xgb":
        pre1 = make_preprocessor(scale=False, num_cols_arg=ns1, cat_cols_arg=cas1)
        Xtr1 = pre1.fit_transform(X_train[cs1])
        Xva1 = pre1.transform(X_val[cs1])
        m1 = XGBClassifier(
            n_estimators=int(bp_s1.get("n_estimators", 330)),
            max_depth=int(bp_s1.get("max_depth", 4)),
            learning_rate=float(bp_s1.get("learning_rate", 0.050)),
            subsample=float(bp_s1.get("subsample", 0.913)),
            colsample_bytree=float(bp_s1.get("colsample_bytree", 0.718)),
            reg_lambda=float(bp_s1.get("reg_lambda", 3.185)),
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss", verbosity=0,
        )
        fit_kw1 = {}
        if weights is not None:
            fit_kw1["sample_weight"] = weights
        m1.fit(Xtr1, ys_tr, **fit_kw1)
        p_stay_val = m1.predict_proba(Xva1)[:, 1]

        pre2 = make_preprocessor(scale=False, num_cols_arg=ns2, cat_cols_arg=cas2)
        pre2.fit(X_train[cs2])
        Xtr2 = pre2.transform(X_train[cs2])[churned_tr]
        Xva2 = pre2.transform(X_val[cs2])
        y_tr2 = yv_tr[churned_tr]
        fit_kw2 = {}
        if weights is not None:
            fit_kw2["sample_weight"] = weights[churned_tr]
        m2 = XGBClassifier(
            n_estimators=int(bp_s2.get("n_estimators", 394)),
            max_depth=int(bp_s2.get("max_depth", 7)),
            learning_rate=float(bp_s2.get("learning_rate", 0.020)),
            subsample=float(bp_s2.get("subsample", 0.883)),
            colsample_bytree=float(bp_s2.get("colsample_bytree", 0.700)),
            reg_lambda=float(bp_s2.get("reg_lambda", 4.322)),
            random_state=RANDOM_STATE + 1, n_jobs=-1, eval_metric="logloss", verbosity=0,
        )
        m2.fit(Xtr2, y_tr2, **fit_kw2)
        p_vol_if_churn_val = m2.predict_proba(Xva2)[:, 1]
    else:
        raise ValueError(mode)

    y_churn_val = (ys_va == 0).astype(np.int8)
    p_churn_val = 1.0 - np.asarray(p_stay_val, dtype=np.float64)
    s1a, s1f, s1r = _cls_metrics(y_churn_val, p_churn_val)
    diag["stage1_acc"] = s1a
    diag["stage1_f1"] = s1f
    diag["stage1_auc"] = s1r
    mask_churn_val = ys_va == 0
    if mask_churn_val.sum() > 0:
        s2a, s2f, s2r = _cls_metrics(yv_va[mask_churn_val], p_vol_if_churn_val[mask_churn_val])
        diag["stage2_acc"] = s2a
        diag["stage2_f1"] = s2f
        diag["stage2_auc"] = s2r
    else:
        diag["stage2_acc"] = diag["stage2_f1"] = diag["stage2_auc"] = float("nan")

    proba3 = hierarchical_class_proba(p_stay_val, p_vol_if_churn_val)
    return proba3, diag


def run_experiment(
    label,
    X_exp=None,
    cols_s1=None, cols_s2=None,
    num_s1=None, cat_s1=None,
    num_s2=None, cat_s2=None,
    sample_weight_fn=None,
    train_filter_fn=None,
    modes=("catboost", "xgb"),
):
    X_use = X_exp if X_exp is not None else X_df
    cs1 = cols_s1 if cols_s1 is not None else cols_stage1
    cs2 = cols_s2 if cols_s2 is not None else cols_stage2
    ns1 = num_s1 if num_s1 is not None else num_stage1
    cas1 = cat_s1 if cat_s1 is not None else cat_stage1
    ns2 = num_s2 if num_s2 is not None else num_stage2
    cas2 = cat_s2 if cat_s2 is not None else cat_stage2

    tss = TimeSeriesSplit(n_splits=N_TS_SPLITS)
    rows = []
    for fold, (tr, va) in enumerate(tss.split(X_use)):
        if train_filter_fn is not None:
            tr = train_filter_fn(tr)
        sw = sample_weight_fn(tr) if sample_weight_fn is not None else None
        X_tr, X_va = X_use.iloc[tr], X_use.iloc[va]
        ys_tr, ys_va = y_binary_stay[tr], y_binary_stay[va]
        yv_tr, yv_va = y_churn_subtype[tr], y_churn_subtype[va]
        churned_tr = (df["churn_status"].iloc[tr] != "not_churned").values

        for mode in modes:
            proba, diag = _fit_predict_twostage(
                X_tr, X_va, ys_tr, ys_va, yv_tr, yv_va,
                churned_tr, mode, cs1, cs2, ns1, cas1, ns2, cas2, sw,
            )
            m = metrics_binary_churn_pct(ys_va, proba)
            m.update(diag)
            m["fold"] = fold
            m["model"] = mode
            rows.append(m)

    result = pd.DataFrame(rows)
    result["experiment"] = label
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for mode in modes:
        sub = result[result["model"] == mode]
        print(f"  [{mode:>8s}]  acc={sub['accuracy_pct'].mean():.2f}  "
              f"f1={sub['f1_churn_pct'].mean():.2f}  "
              f"auc={sub['roc_auc_churn_pct'].mean():.2f}  |  "
              f"s2_f1={sub['stage2_f1'].mean():.2f}  "
              f"s2_auc={sub['stage2_auc'].mean():.2f}")
    all_experiment_results.append(result)
    return result

print("Experiment runner ready.")


# === CELL 4 ===
baseline = run_experiment("0_baseline")

# === CELL 5 ===
def temporal_weights_linear(train_idx):
    return np.linspace(0.5, 1.5, len(train_idx))

exp1 = run_experiment("1_sample_weight_linear", sample_weight_fn=temporal_weights_linear)


# === CELL 6 ===
def temporal_weights_exp(train_idx):
    n = len(train_idx)
    raw = np.exp(np.linspace(0, 1.5, n))
    return raw / raw.mean()

exp1b = run_experiment("1b_sample_weight_exp", sample_weight_fn=temporal_weights_exp)


# === CELL 7 ===
drop = {"country_code"}
exp2_cs1 = [c for c in cols_stage1 if c not in drop]
exp2_cs2 = [c for c in cols_stage2 if c not in drop]
exp2_cas1 = [c for c in cat_stage1 if c not in drop]
exp2_cas2 = [c for c in cat_stage2 if c not in drop]

exp2 = run_experiment(
    "2_drop_country",
    cols_s1=exp2_cs1, cols_s2=exp2_cs2,
    cat_s1=exp2_cas1, cat_s2=exp2_cas2,
)


# === CELL 8 ===
X_exp3 = X_df.copy()
X_exp3["avg_purchase"] = X_exp3["purch_amount_sum"] / X_exp3["purch_n"].clip(lower=1)
X_exp3["flags_x_purch"] = X_exp3["att_card_flags_mean"] * X_exp3["purch_n"]
X_exp3["log_purch_amount"] = np.log1p(X_exp3["purch_amount_sum"].fillna(0).clip(lower=0))
X_exp3["frustration_x_plan"] = (
    X_exp3["frustration"].fillna("skipped").astype(str) + "_"
    + X_exp3["usage_plan"].fillna("skipped").astype(str)
)

new_num = ["avg_purchase", "flags_x_purch", "log_purch_amount"]
new_cat = ["frustration_x_plan"]
exp3_num_s1 = num_stage1 + new_num
exp3_cat_s1 = cat_stage1 + new_cat
exp3_cs1 = cols_stage1 + new_num + new_cat
exp3_num_s2 = num_stage2 + new_num
exp3_cat_s2 = cat_stage2 + new_cat
exp3_cs2 = cols_stage2 + new_num + new_cat

exp3 = run_experiment(
    "3_interactions",
    X_exp=X_exp3,
    cols_s1=exp3_cs1, cols_s2=exp3_cs2,
    num_s1=exp3_num_s1, cat_s1=exp3_cat_s1,
    num_s2=exp3_num_s2, cat_s2=exp3_cat_s2,
)


# === CELL 9 ===
def late_only_50(train_idx):
    n = len(train_idx)
    return train_idx[n // 2:]

exp4 = run_experiment("4_late_only_50pct", train_filter_fn=late_only_50)


# === CELL 10 ===
ALPHA = 0.4  # weight for full model

tss = TimeSeriesSplit(n_splits=N_TS_SPLITS)
rows_e5 = []
for fold, (tr, va) in enumerate(tss.split(X_df)):
    tr_late = tr[len(tr) // 2:]
    X_tr_full, X_tr_late, X_va = X_df.iloc[tr], X_df.iloc[tr_late], X_df.iloc[va]
    ys_full = y_binary_stay[tr]
    ys_late = y_binary_stay[tr_late]
    ys_va = y_binary_stay[va]
    yv_full = y_churn_subtype[tr]
    yv_late = y_churn_subtype[tr_late]
    yv_va = y_churn_subtype[va]
    churned_full = (df["churn_status"].iloc[tr] != "not_churned").values
    churned_late = (df["churn_status"].iloc[tr_late] != "not_churned").values

    for mode in ("catboost", "xgb"):
        p3_full, _ = _fit_predict_twostage(
            X_tr_full, X_va, ys_full, ys_va, yv_full, yv_va,
            churned_full, mode,
            cols_stage1, cols_stage2, num_stage1, cat_stage1, num_stage2, cat_stage2,
        )
        p3_late, _ = _fit_predict_twostage(
            X_tr_late, X_va, ys_late, ys_va, yv_late, yv_va,
            churned_late, mode,
            cols_stage1, cols_stage2, num_stage1, cat_stage1, num_stage2, cat_stage2,
        )
        p3_blend = ALPHA * p3_full + (1 - ALPHA) * p3_late

        m = metrics_binary_churn_pct(ys_va, p3_blend)
        y_churn = (ys_va == 0).astype(np.int8)
        p_churn = p3_blend[:, 0] + p3_blend[:, 1]
        s1a, s1f, s1r = _cls_metrics(y_churn, p_churn)
        m["stage1_acc"] = s1a
        m["stage1_f1"] = s1f
        m["stage1_auc"] = s1r
        mask_c = ys_va == 0
        if mask_c.sum() > 0:
            p_vol_blend = p3_blend[mask_c, 1] / (p3_blend[mask_c, 0] + p3_blend[mask_c, 1]).clip(1e-8)
            s2a, s2f, s2r = _cls_metrics(yv_va[mask_c], p_vol_blend)
            m["stage2_acc"] = s2a
            m["stage2_f1"] = s2f
            m["stage2_auc"] = s2r
        else:
            m["stage2_acc"] = m["stage2_f1"] = m["stage2_auc"] = float("nan")
        m["fold"] = fold
        m["model"] = mode
        rows_e5.append(m)

exp5 = pd.DataFrame(rows_e5)
exp5["experiment"] = "5_temporal_ensemble"
all_experiment_results.append(exp5)

print(f"\n{'='*60}")
print(f"  5_temporal_ensemble (alpha={ALPHA})")
print(f"{'='*60}")
for mode in ("catboost", "xgb"):
    sub = exp5[exp5["model"] == mode]
    print(f"  [{mode:>8s}]  acc={sub['accuracy_pct'].mean():.2f}  "
          f"f1={sub['f1_churn_pct'].mean():.2f}  "
          f"auc={sub['roc_auc_churn_pct'].mean():.2f}  |  "
          f"s2_f1={sub['stage2_f1'].mean():.2f}  "
          f"s2_auc={sub['stage2_auc'].mean():.2f}")


# === CELL 11 ===
SMOOTH = 20  # smoothing factor

drop_te = {"country_code"}
te_cs1 = [c for c in cols_stage1 if c not in drop_te] + ["country_churn_rate"]
te_cs2 = [c for c in cols_stage2 if c not in drop_te] + ["country_churn_rate"]
te_ns1 = [c for c in num_stage1] + ["country_churn_rate"]
te_cas1 = [c for c in cat_stage1 if c not in drop_te]
te_ns2 = [c for c in num_stage2] + ["country_churn_rate"]
te_cas2 = [c for c in cat_stage2 if c not in drop_te]

tss = TimeSeriesSplit(n_splits=N_TS_SPLITS)
rows_e6 = []
for fold, (tr, va) in enumerate(tss.split(X_df)):
    X_tr = X_df.iloc[tr].copy()
    X_va = X_df.iloc[va].copy()
    ys_tr = y_binary_stay[tr]
    ys_va = y_binary_stay[va]
    yv_tr = y_churn_subtype[tr]
    yv_va = y_churn_subtype[va]
    churned_tr = (df["churn_status"].iloc[tr] != "not_churned").values

    y_churn_train = (ys_tr == 0).astype(np.float64)
    global_prior = y_churn_train.mean()
    cc_train = X_tr["country_code"].values
    te_map = {}
    for code_val in np.unique(cc_train):
        mask = cc_train == code_val
        cnt = mask.sum()
        rate = y_churn_train[mask].mean()
        te_map[code_val] = (cnt * rate + SMOOTH * global_prior) / (cnt + SMOOTH)

    X_tr["country_churn_rate"] = X_tr["country_code"].map(te_map).fillna(global_prior).astype(np.float64)
    X_va["country_churn_rate"] = X_va["country_code"].map(te_map).fillna(global_prior).astype(np.float64)

    for mode in ("catboost", "xgb"):
        proba, diag = _fit_predict_twostage(
            X_tr, X_va, ys_tr, ys_va, yv_tr, yv_va,
            churned_tr, mode,
            te_cs1, te_cs2, te_ns1, te_cas1, te_ns2, te_cas2,
        )
        m = metrics_binary_churn_pct(ys_va, proba)
        m.update(diag)
        m["fold"] = fold
        m["model"] = mode
        rows_e6.append(m)

exp6 = pd.DataFrame(rows_e6)
exp6["experiment"] = "6_target_encode_country"
all_experiment_results.append(exp6)

print(f"\n{'='*60}")
print(f"  6_target_encode_country (smooth={SMOOTH})")
print(f"{'='*60}")
for mode in ("catboost", "xgb"):
    sub = exp6[exp6["model"] == mode]
    print(f"  [{mode:>8s}]  acc={sub['accuracy_pct'].mean():.2f}  "
          f"f1={sub['f1_churn_pct'].mean():.2f}  "
          f"auc={sub['roc_auc_churn_pct'].mean():.2f}  |  "
          f"s2_f1={sub['stage2_f1'].mean():.2f}  "
          f"s2_auc={sub['stage2_auc'].mean():.2f}")


# === CELL 12 ===
import matplotlib.pyplot as plt

full = pd.concat(all_experiment_results, ignore_index=True)

agg_cols = ["accuracy_pct", "f1_churn_pct", "roc_auc_churn_pct",
            "stage1_acc", "stage1_f1", "stage1_auc",
            "stage2_acc", "stage2_f1", "stage2_auc"]
summary = full.groupby(["experiment", "model"])[agg_cols].agg(["mean", "std"])

print("=" * 80)
print("  FULL COMPARISON TABLE (mean +/- std across 5 folds)")
print("=" * 80)
display(summary)

key_cols = ["accuracy_pct", "f1_churn_pct", "roc_auc_churn_pct"]
compact = full.groupby(["experiment", "model"])[key_cols].mean().round(2)
print("\nCompact mean metrics:")
display(compact)

for mode in ("catboost", "xgb"):
    sub = full[full["model"] == mode]
    pivot = sub.groupby("experiment")[key_cols].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{mode.upper()} — Experiment Comparison", fontsize=14)
    for ax, col in zip(axes, key_cols):
        pivot[col].plot.barh(ax=ax)
        ax.set_xlabel(col)
        ax.axvline(pivot.loc["0_baseline", col], color="red", linestyle="--", alpha=0.7, label="baseline")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

print("\nDelta vs baseline (positive = better):")
for mode in ("catboost", "xgb"):
    sub = full[full["model"] == mode]
    pivot = sub.groupby("experiment")[key_cols].mean()
    base_row = pivot.loc["0_baseline"]
    delta = pivot.subtract(base_row).drop("0_baseline")
    print(f"\n  [{mode}]")
    display(delta.round(3))


# === CELL 13 ===
X_combo = X_df.copy()
X_combo["avg_purchase"] = X_combo["purch_amount_sum"] / X_combo["purch_n"].clip(lower=1)
X_combo["flags_x_purch"] = X_combo["att_card_flags_mean"] * X_combo["purch_n"]
X_combo["log_purch_amount"] = np.log1p(X_combo["purch_amount_sum"].fillna(0).clip(lower=0))
X_combo["frustration_x_plan"] = (
    X_combo["frustration"].fillna("skipped").astype(str) + "_"
    + X_combo["usage_plan"].fillna("skipped").astype(str)
)

drop_c = {"country_code"}
new_num_c = ["avg_purchase", "flags_x_purch", "log_purch_amount"]
new_cat_c = ["frustration_x_plan"]
combo_cs1 = [c for c in cols_stage1 if c not in drop_c] + new_num_c + new_cat_c
combo_cs2 = [c for c in cols_stage2 if c not in drop_c] + new_num_c + new_cat_c
combo_ns1 = [c for c in num_stage1 if c not in drop_c] + new_num_c
combo_cas1 = [c for c in cat_stage1 if c not in drop_c] + new_cat_c
combo_ns2 = [c for c in num_stage2 if c not in drop_c] + new_num_c
combo_cas2 = [c for c in cat_stage2 if c not in drop_c] + new_cat_c

combo = run_experiment(
    "7_combo_interact_drop_weight",
    X_exp=X_combo,
    cols_s1=combo_cs1, cols_s2=combo_cs2,
    num_s1=combo_ns1, cat_s1=combo_cas1,
    num_s2=combo_ns2, cat_s2=combo_cas2,
    sample_weight_fn=temporal_weights_linear,
)


# === CELL 14 ===
full2 = pd.concat(all_experiment_results, ignore_index=True)

key_cols2 = ["accuracy_pct", "f1_churn_pct", "roc_auc_churn_pct"]
print("=" * 80)
print("  FINAL RANKING by roc_auc_churn_pct (higher = better)")
print("=" * 80)
for mode in ("catboost", "xgb"):
    sub = full2[full2["model"] == mode]
    ranking = sub.groupby("experiment")[key_cols2].mean().sort_values("roc_auc_churn_pct", ascending=False)
    print(f"\n  [{mode}]")
    display(ranking.round(3))


