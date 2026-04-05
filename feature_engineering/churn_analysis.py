"""
feature_engineering.py
=======================
Builds a flat per-user feature matrix from all Higgsfield tables.

**Vol/invol intent signals (screenshots / deck):** implemented as ``inter_*`` columns in
``build_interaction_features`` — payment × geo (``inter_3ds_x_geohigh``,
``inter_cvc_unav_x_geohigh``, …), stacked friction ``inter_payment_risk_score`` (0–4:
prepaid + cvc_unavailable + 3ds + billing≠card), quiz/geo (``inter_jp_x_jcb``,
``inter_update_x_no_credits``, ``inter_inconsistent_x_vol_geo``). Use
``build_inter_features_without_generations()`` to compute them **without** loading
``train_users_generations.csv`` (avoids OOM). Full ``build_features(..., generations_path=)``
adds generation-based ``inter_*`` only when you need those extras.

**experiment_v2:** ``--churn-analysis-inter`` uses ``build_inter_features_without_generations``,
not the full generations pipeline.

Usage:
    import feature_engineering as fe
    df = fe.build_features(
        users_path       = "data/preprocessed/train/train_users.csv",
        props_path       = "data/preprocessed/train/train_users_properties.csv",
        purchases_path   = "data/preprocessed/train/train_users_purchases.csv",
        txn_path         = "data/preprocessed/train/train_users_transaction_attempts.csv",
        quizzes_path     = "data/preprocessed/train/train_users_quizzes.csv",
        generations_path = "data/preprocessed/train/train_users_generations.csv",  # optional
        is_train         = True,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# GEO RISK MAPS  (from empirical analysis of train set)
# ──────────────────────────────────────────────────────────────────────────────

# Countries where invol_churn >> vol_churn (payment infra issues)
_INVOL_RISK_COUNTRIES = frozenset({
    "in", "tr", "br", "kh", "hk", "vn", "az", "my", "ph", "mx",
    "tg", "aq", "cf", "vu", "gd", "mo", "rw", "ug",
})

# Countries where vol_churn >> invol_churn (product-market fit issues)
_VOL_RISK_COUNTRIES = frozenset({
    "jp", "de", "fr", "kr",
})

# Country-level payment failure rates from txn_attempts analysis
# (billing_country → approx failure rate, for use as a continuous feature)
_COUNTRY_FAIL_RATE = {
    "kh": 0.877, "mo": 0.876, "aq": 0.859, "tg": 0.844, "jp": 0.751,
    "hk": 0.741, "my": 0.735, "vn": 0.701, "az": 0.692, "bo": 0.626,
    "ph": 0.550, "tr": 0.480, "in": 0.460, "br": 0.380, "mx": 0.350,
    "fr": 0.280, "kr": 0.250, "de": 0.220, "us": 0.200, "gb": 0.180,
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_mode(series: pd.Series) -> str:
    m = series.dropna().mode()
    return m.iloc[0] if len(m) > 0 else "unknown"


def _plan_tier(plan: str) -> int:
    return {
        "Higgsfield Basic": 1,
        "Higgsfield Creator": 2,
        "Higgsfield Pro": 3,
        "Higgsfield Ultimate": 4,
    }.get(plan, 2)


# ──────────────────────────────────────────────────────────────────────────────
# TABLE LOADERS
# ──────────────────────────────────────────────────────────────────────────────

def _load(path: str | Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, **kwargs)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDERS (one per source table)
# ──────────────────────────────────────────────────────────────────────────────

def build_purchase_features(purchases: pd.DataFrame) -> pd.DataFrame:
    """Per-user purchase behaviour features."""
    base = purchases.groupby("user_id").agg(
        n_purchases=("transaction_id", "count"),
        total_spend=("purchase_amount_dollars", "sum"),
        max_spend=("purchase_amount_dollars", "max"),
        n_purchase_types=("purchase_type", "nunique"),
    ).reset_index()

    for ptype, col in [
        ("Credits package",     "has_credits"),
        ("Upsell",              "has_upsell"),
        ("Subscription Update", "has_update"),
        ("Reactivation",        "has_reactivation"),
        ("Gift",                "has_gift"),
    ]:
        flag = (
            purchases[purchases["purchase_type"] == ptype][["user_id"]]
            .drop_duplicates()
            .assign(**{col: 1})
        )
        base = base.merge(flag, on="user_id", how="left")
        base[col] = base[col].fillna(0).astype(int)

    # Derived
    base["avg_spend_per_purchase"] = (base["total_spend"] / base["n_purchases"]).fillna(0)
    # Spending above subscription price → bought extras
    base["spent_above_35"] = (base["total_spend"] > 35).astype(int)

    return base


def build_card_features(txn: pd.DataFrame, purchases: pd.DataFrame) -> pd.DataFrame:
    """Per-user card / payment method features derived from txn_attempts."""
    linked = txn.merge(purchases[["user_id", "transaction_id"]], on="transaction_id", how="left")

    cat_agg = linked.groupby("user_id").agg(
        card_brand=("card_brand",   _safe_mode),
        cvc_check= ("cvc_check",    _safe_mode),
        card_country=("card_country", _safe_mode),
        card_funding=("card_funding", _safe_mode),
        bank_country=("bank_country", _safe_mode),
    ).reset_index()

    bool_agg = linked.groupby("user_id").agg(
        is_prepaid=  ("is_prepaid",   lambda x: int((x.astype(str).str.lower() == "true").any())),
        is_virtual=  ("is_virtual",   lambda x: int((x.astype(str).str.lower() == "true").any())),
        is_business= ("is_business",  lambda x: int((x.astype(str).str.lower() == "true").any())),
        is_3d_secure=("is_3d_secure", lambda x: int((x.astype(str).str.lower() == "true").any())),
        is_3d_auth=  ("is_3d_secure_authenticated",
                      lambda x: int((x.astype(str).str.lower() == "true").any())),
        uses_apple_pay=  ("digital_wallet", lambda x: int("apple_pay" in x.values)),
        uses_android_pay=("digital_wallet", lambda x: int("android_pay" in x.values)),
        n_txn_attempts=  ("transaction_id", "count"),
        n_3d_secure_used=("is_3d_secure",   lambda x: (x.astype(str).str.lower() == "true").sum()),
    ).reset_index()

    out = cat_agg.merge(bool_agg, on="user_id", how="left")

    # Derived binary flags
    out["is_jcb"]          = (out["card_brand"] == "jcb").astype(int)
    out["is_amex"]         = (out["card_brand"] == "amex").astype(int)
    out["is_debit"]        = (out["card_funding"] == "debit").astype(int)
    out["cvc_unavailable"] = (out["cvc_check"] == "unavailable").astype(int)
    out["cvc_fail"]        = (out["cvc_check"] == "fail").astype(int)
    out["uses_digital_wallet"] = (out["uses_apple_pay"] | out["uses_android_pay"]).astype(int)

    return out


def build_property_features(props: pd.DataFrame) -> pd.DataFrame:
    """Per-user subscription property features."""
    out = props[["user_id", "subscription_plan", "country_code"]].copy()
    out["plan_tier"]  = out["subscription_plan"].apply(_plan_tier)
    out["is_ultimate"]= (out["subscription_plan"] == "Higgsfield Ultimate").astype(int)
    out["is_basic"]   = (out["subscription_plan"] == "Higgsfield Basic").astype(int)
    return out


def build_quiz_features(quizzes: pd.DataFrame) -> pd.DataFrame:
    """Per-user quiz / onboarding features."""
    # Take first row per user (quiz may have duplicates in raw data)
    q = quizzes.drop_duplicates("user_id").copy()
    cols = ["user_id", "source", "experience", "usage_plan",
            "frustration", "first_feature", "role", "team_size"]
    q = q[[c for c in cols if c in q.columns]]

    # Binary intent signals
    q["from_chatgpt"]   = (q["source"] == "chatgpt").astype(int)
    q["from_friends"]   = (q["source"].isin(["friends", "friend"])).astype(int)
    q["from_instagram"] = (q["source"] == "instagram").astype(int)
    q["from_tiktok"]    = (q["source"] == "tiktok").astype(int)

    q["is_beginner"]    = (q["experience"] == "beginner").astype(int)
    q["is_expert"]      = (q["experience"].isin(["expert", "advanced"])).astype(int)

    q["frustration_highcost"]    = (q["frustration"] == "high-cost").astype(int)
    q["frustration_inconsistent"]= (q["frustration"] == "inconsistent").astype(int)
    q["frustration_limited"]     = (q["frustration"] == "limited").astype(int)
    q["frustration_confusing"]   = (q["frustration"] == "confusing").astype(int)
    q["frustration_hardprompt"]  = (q["frustration"] == "hard-prompt").astype(int)

    q["usage_personal"]  = (q["usage_plan"] == "personal").astype(int)
    q["usage_marketing"] = (q["usage_plan"] == "marketing").astype(int)
    q["usage_filmmaking"]= (q["usage_plan"] == "filmmaking").astype(int)
    q["usage_social"]    = (q["usage_plan"] == "social").astype(int)

    q["is_solo"] = q["team_size"].astype(str).isin(["1", "solo", "solo_1"]).astype(int)
    return q


def build_generation_features(generations: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user generation behaviour features from the first 14 days.

    Assumes columns: user_id, created_at (or generation_time), generation_type (optional).
    Adjust column names to match actual schema.
    """
    # --- Normalise column names (adjust if actual names differ) ---
    time_col = next(
        (c for c in generations.columns if "time" in c.lower() or "created" in c.lower() or "date" in c.lower()),
        None,
    )
    type_col = next(
        (c for c in generations.columns if "type" in c.lower() or "model" in c.lower() or "mode" in c.lower()),
        None,
    )

    gen = generations.copy()
    if time_col:
        gen["_ts"] = pd.to_datetime(gen[time_col], utc=True, errors="coerce")
    else:
        gen["_ts"] = pd.NaT

    # --- User-level aggregates ---
    agg = gen.groupby("user_id").agg(
        total_generations=("user_id", "count"),
    ).reset_index()

    if time_col and gen["_ts"].notna().any():
        time_agg = gen.groupby("user_id")["_ts"].agg(
            first_gen_ts="min",
            last_gen_ts="max",
        ).reset_index()
        agg = agg.merge(time_agg, on="user_id", how="left")

        # Subscription start date for lag calculation (merged externally if needed)
        # Here we compute day-of-generation distribution
        gen["_day"] = gen.groupby("user_id")["_ts"].transform(
            lambda x: (x - x.min()).dt.days
        )
        day_agg = gen.groupby("user_id")["_day"].agg(
            gen_last_day="max",         # last active day within 14d window
            gen_first_day="min",        # 0 if generated on signup day
            gen_day_spread=lambda x: x.max() - x.min(),  # span of activity
        ).reset_index()
        agg = agg.merge(day_agg, on="user_id", how="left")

        # Days with at least one generation (active days)
        active_days = gen.groupby("user_id")["_day"].nunique().reset_index()
        active_days.columns = ["user_id", "n_active_gen_days"]
        agg = agg.merge(active_days, on="user_id", how="left")

        # Gens per active day
        agg["gen_per_active_day"] = (
            agg["total_generations"] / agg["n_active_gen_days"]
        ).replace([np.inf], np.nan)

        # Activated on day 0 (generated on signup day — strong retention signal)
        day0 = gen[gen["_day"] == 0][["user_id"]].drop_duplicates().assign(activated_day0=1)
        agg = agg.merge(day0, on="user_id", how="left")
        agg["activated_day0"] = agg["activated_day0"].fillna(0).astype(int)

        # Late activation (first gen on day 7+) — risk signal
        agg["late_activator"] = (agg["gen_first_day"] >= 7).astype(int)

        # Trend: gens in first 7 days vs last 7 days (day 0-6 vs 7-13)
        gen["_half"] = (gen["_day"] >= 7).astype(int)
        half_agg = gen.groupby(["user_id", "_half"]).size().unstack(fill_value=0).reset_index()
        half_agg.columns = ["user_id", "gen_first_half", "gen_second_half"]
        agg = agg.merge(half_agg, on="user_id", how="left")
        agg["gen_trend_ratio"] = (
            (agg["gen_second_half"] + 1) / (agg["gen_first_half"] + 1)
        )  # >1 = growing, <1 = declining

    if type_col:
        type_agg = gen.groupby("user_id")[type_col].agg(
            n_gen_types="nunique",
            top_gen_type=_safe_mode,
        ).reset_index()
        agg = agg.merge(type_agg, on="user_id", how="left")

    # Power-user threshold: 20+ gens in 14 days
    agg["gen_power_user"] = (agg["total_generations"] >= 20).astype(int)
    # Inactive: 0 gens (no engagement — high churn risk)
    agg["gen_zero"] = (agg["total_generations"] == 0).astype(int)

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# INTERACTION FEATURE BUILDER  ← the core of this module
# ──────────────────────────────────────────────────────────────────────────────

def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-column interaction features.
    Input df must have been merged from all source tables already.

    All features are documented with their empirical lift from the train set:
    - invol_vs_vol = invol_churn_rate - vol_churn_rate for value=1
    - invol_base   = invol_churn_rate - 0.25 (baseline)
    - vol_base     = vol_churn_rate - 0.25 (baseline)
    """
    out = df.copy()

    cc = out.get("country_code", pd.Series("", index=out.index)).str.lower().fillna("")
    cd = out.get("card_country",  pd.Series("", index=out.index)).str.lower().fillna("")

    def _flag(series, name):
        out[name] = series.astype(int)

    # ── PAYMENT CARD SIGNALS ──────────────────────────────────────────────────

    # CVC unavailable × geo risk  [invol_vs_vol=+0.300 ← strongest single signal]
    _flag(
        out.get("cvc_unavailable", 0).fillna(0).astype(int) &
        cc.isin(_INVOL_RISK_COUNTRIES).astype(int),
        "inter_cvc_unav_x_geohigh",
    )

    # 3D-Secure × invol-risk country  [invol_vs_vol=+0.300]
    _flag(
        out.get("is_3d_secure", 0).fillna(0).astype(int) &
        cc.isin(_INVOL_RISK_COUNTRIES).astype(int),
        "inter_3ds_x_geohigh",
    )

    # Prepaid × invol-risk country  [invol_vs_vol=+0.184]
    _flag(
        out.get("is_prepaid", 0).fillna(0).astype(int) &
        cc.isin(_INVOL_RISK_COUNTRIES).astype(int),
        "inter_prepaid_x_geohigh",
    )

    # Debit × invol-risk country  [invol_vs_vol=+0.164]
    _flag(
        out.get("is_debit", 0).fillna(0).astype(int) &
        cc.isin(_INVOL_RISK_COUNTRIES).astype(int),
        "inter_debit_x_geohigh",
    )

    # Billing ≠ card country × invol-risk geo  [invol_vs_vol=+0.046]
    country_mismatch = (cc != cd).astype(int)
    out["inter_mismatch"] = country_mismatch
    _flag(
        country_mismatch & cc.isin(_INVOL_RISK_COUNTRIES).astype(int),
        "inter_mismatch_x_geohigh",
    )

    # Payment risk composite score (0–4):
    #   score=3 → invol=0.658  vol=0.116  [invol_vs_vol=+0.542 ← extreme]
    #   score=2 → invol=0.442  vol=0.273
    #   score=1 → invol=0.296  vol=0.248
    #   score=0 → invol=0.210  vol=0.251
    out["inter_payment_risk_score"] = (
        out.get("is_prepaid",     0).fillna(0).astype(int) +
        out.get("cvc_unavailable",0).fillna(0).astype(int) +
        out.get("is_3d_secure",   0).fillna(0).astype(int) +
        country_mismatch
    )

    # Country-level failure rate (continuous)  [use _COUNTRY_FAIL_RATE lookup]
    out["inter_country_fail_rate"] = cc.map(_COUNTRY_FAIL_RATE).fillna(0.25)

    # ── VOLUNTARY CHURN SIGNALS ───────────────────────────────────────────────

    # JCB card  [vol=0.615  invol=0.365  vol_vs_vol_base=+0.365]
    # Already computed as is_jcb — expose as interaction name for clarity
    out["inter_jcb_card"] = out.get("is_jcb", (out.get("card_brand","") == "jcb")).astype(int)

    # Japan country × JCB card  [vol=0.595  invol=0.390  vol_base=+0.345]
    _flag(
        (cc == "jp").astype(int) & out["inter_jcb_card"],
        "inter_jp_x_jcb",
    )

    # Subscription update but no credits bought  [vol=0.417  invol=0.238  vol_base=+0.167]
    # Interpretation: explored plan change but didn't commit with extra credit spend
    _flag(
        out.get("has_update", 0).fillna(0).astype(int) &
        (1 - out.get("has_credits", 0).fillna(0).astype(int)),
        "inter_update_x_no_credits",
    )

    # Vol-risk country × subscription update (churner who tried to stay)
    _flag(
        cc.isin(_VOL_RISK_COUNTRIES).astype(int) &
        out.get("has_update", 0).fillna(0).astype(int),
        "inter_vol_geo_x_update",
    )

    # High plan tier × high-cost frustration  [vol_base=+0.035]
    _flag(
        (out.get("frustration", "") == "high-cost").astype(int) &
        (out.get("plan_tier", 2) >= 3).astype(int),
        "inter_highcost_x_highplan",
    )

    # ChatGPT source × expert/advanced experience (power-user comparison shopper)
    _flag(
        (out.get("source", "") == "chatgpt").astype(int) &
        out.get("experience", "").isin(["expert", "advanced"]).astype(int),
        "inter_chatgpt_x_expert",
    )

    # Inconsistent frustration × vol-risk country  [vol_base=+0.025]
    _flag(
        (out.get("frustration", "") == "inconsistent").astype(int) &
        cc.isin(_VOL_RISK_COUNTRIES).astype(int),
        "inter_inconsistent_x_vol_geo",
    )

    # Upsell buyer in vol-risk country (price-sensitive but explored more)
    _flag(
        out.get("has_upsell", 0).fillna(0).astype(int) &
        cc.isin(_VOL_RISK_COUNTRIES).astype(int),
        "inter_upsell_x_vol_geo",
    )

    # ── RETENTION SIGNALS (invol AND vol both low) ────────────────────────────

    # Credits + Apple Pay  [retain=0.662 ← highest single retention signal]
    _flag(
        out.get("has_credits", 0).fillna(0).astype(int) &
        out.get("uses_apple_pay", 0).fillna(0).astype(int),
        "inter_credits_x_apple_pay",
    )

    # Marketing usage_plan × any purchase (stickiest segment)
    _flag(
        (out.get("usage_plan", "") == "marketing").astype(int) &
        (out.get("n_purchases", 0).fillna(0) > 1).astype(int),
        "inter_marketing_x_multipurchase",
    )

    # Friends referral × no payment issues (organic + stable payment)
    _flag(
        out.get("from_friends", (out.get("source","").isin(["friends","friend"]))).astype(int) &
        (out["inter_payment_risk_score"] == 0).astype(int),
        "inter_friends_x_no_paymentrisk",
    )

    # ── GENERATION × PAYMENT INTERACTIONS (use if gen features available) ─────

    if "total_generations" in out.columns:
        gen_active = (out["total_generations"] > 0).astype(int)
        gen_high   = (out["total_generations"] >= 20).astype(int)

        # Active user + payment risk → invol churn despite engagement
        _flag(
            gen_active & (out["inter_payment_risk_score"] >= 2).astype(int),
            "inter_gen_active_x_high_payment_risk",
        )

        # Zero generations + vol-risk country → disengaged + likely vol churn
        _flag(
            (out.get("gen_zero", (out["total_generations"] == 0)).astype(int)) &
            cc.isin(_VOL_RISK_COUNTRIES).astype(int),
            "inter_zero_gen_x_vol_geo",
        )

        # Power user + smooth payment (digital wallet) → strongest retention
        _flag(
            gen_high & out.get("uses_digital_wallet", out.get("uses_apple_pay", 0)).fillna(0).astype(int),
            "inter_power_user_x_smooth_pay",
        )

        # Activated day 0 + bought credits → ultra-sticky
        if "activated_day0" in out.columns:
            _flag(
                out["activated_day0"] & out.get("has_credits", 0).fillna(0).astype(int),
                "inter_day0_activation_x_credits",
            )

        # Declining generation trend + vol-risk country (fading interest)
        if "gen_trend_ratio" in out.columns:
            out["inter_gen_decline_x_vol_geo"] = (
                (out["gen_trend_ratio"] < 0.7).astype(int) &
                cc.isin(_VOL_RISK_COUNTRIES).astype(int)
            ).astype(int)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# CORE TABLE MERGE (no generations)
# ──────────────────────────────────────────────────────────────────────────────


def load_core_user_tables(
    users_path: str | Path,
    props_path: str | Path,
    purchases_path: str | Path,
    txn_path: str | Path,
    quizzes_path: str | Path,
) -> pd.DataFrame:
    """Load users + properties + purchases + txn + quizzes. Does **not** read generations."""
    print("Loading tables (users, properties, purchases, txn, quizzes)...")
    users = _load(users_path)
    props = _load(props_path)
    purchases = _load(purchases_path)
    txn = _load(txn_path)
    quizzes = _load(quizzes_path)

    print("Building purchase features...")
    pf = build_purchase_features(purchases)

    print("Building card/payment features...")
    cf = build_card_features(txn, purchases)

    print("Building property features...")
    ppf = build_property_features(props)

    print("Building quiz features...")
    qf = build_quiz_features(quizzes)

    df = users.copy()
    df = df.merge(ppf, on="user_id", how="left")
    df = df.merge(pf, on="user_id", how="left")
    df = df.merge(cf, on="user_id", how="left")
    df = df.merge(qf, on="user_id", how="left")

    for c in [
        "n_purchases",
        "total_spend",
        "max_spend",
        "n_purchase_types",
        "has_credits",
        "has_upsell",
        "has_update",
        "has_reactivation",
        "has_gift",
        "avg_spend_per_purchase",
        "spent_above_35",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in [
        "is_jcb",
        "is_amex",
        "is_debit",
        "cvc_unavailable",
        "cvc_fail",
        "is_prepaid",
        "is_virtual",
        "is_business",
        "is_3d_secure",
        "is_3d_auth",
        "uses_apple_pay",
        "uses_android_pay",
        "uses_digital_wallet",
        "n_txn_attempts",
        "n_3d_secure_used",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in ["card_brand", "cvc_check", "card_country", "card_funding", "bank_country"]:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")

    return df


def build_inter_features_without_generations(
    users_path: str | Path,
    props_path: str | Path,
    purchases_path: str | Path,
    txn_path: str | Path,
    quizzes_path: str | Path,
) -> pd.DataFrame:
    """
    Vol/invol ``inter_*`` signals from ``build_interaction_features`` that do not need
    generation aggregates: payment friction (``inter_payment_risk_score``, ``inter_3ds_x_geohigh``,
    ``inter_cvc_unav_x_geohigh``, …), ``inter_jp_x_jcb``, ``inter_update_x_no_credits``,
    ``inter_inconsistent_x_vol_geo``, etc.

    Does **not** load ``train_users_generations.csv`` — avoids OOM when that file is huge.
    """
    df = load_core_user_tables(
        users_path, props_path, purchases_path, txn_path, quizzes_path
    )
    print("Building interaction features (no generations)...")
    return build_interaction_features(df)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def build_features(
    users_path: str | Path,
    props_path: str | Path,
    purchases_path: str | Path,
    txn_path: str | Path,
    quizzes_path: str | Path,
    generations_path: str | Path | None = None,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Build the full per-user feature matrix.

    Returns a DataFrame with one row per user.
    If is_train=True, includes 'churn_status' column.
    """
    df = load_core_user_tables(
        users_path, props_path, purchases_path, txn_path, quizzes_path
    )

    # Optional: generation features
    if generations_path is not None:
        print("Building generation features...")
        gens = _load(generations_path)
        gf = build_generation_features(gens)
        df = df.merge(gf, on="user_id", how="left")
        for c in ["total_generations","n_active_gen_days","activated_day0",
                  "late_activator","gen_power_user","gen_zero",
                  "gen_first_half","gen_second_half"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)
        if "gen_trend_ratio" in df.columns:
            df["gen_trend_ratio"] = df["gen_trend_ratio"].fillna(1.0)

    print("Building interaction features...")
    df = build_interaction_features(df)

    # ── LABEL ENCODING for tree models (LightGBM handles categories natively)
    cat_cols = [
        "subscription_plan", "country_code", "card_brand", "card_funding",
        "cvc_check", "card_country", "bank_country",
        "source", "experience", "usage_plan", "frustration",
        "first_feature", "role", "team_size",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    if is_train and "churn_status" in df.columns:
        label_map = {"not_churned": 0, "vol_churn": 1, "invol_churn": 2}
        df["label"] = df["churn_status"].map(label_map)

    print(f"Feature matrix shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE SUMMARY  (run this to print lift table for all interaction features)
# ──────────────────────────────────────────────────────────────────────────────

def print_interaction_lift_table(df: pd.DataFrame) -> None:
    """Prints a sorted lift table for all inter_* columns vs churn_status."""
    if "churn_status" not in df.columns:
        raise ValueError("Need churn_status column — pass train data.")

    inter_cols = [c for c in df.columns if c.startswith("inter_")]
    rows = []
    for feat in inter_cols:
        col = df[feat].fillna(0)
        for val in [1]:
            mask = col == val
            n = mask.sum()
            if n < 30:
                continue
            invol  = df.loc[mask, "churn_status"].eq("invol_churn").mean()
            vol    = df.loc[mask, "churn_status"].eq("vol_churn").mean()
            retain = df.loc[mask, "churn_status"].eq("not_churned").mean()
            rows.append({
                "feature": feat, "n": n,
                "invol": round(invol, 3), "vol": round(vol, 3), "retain": round(retain, 3),
                "invol-vol": round(invol - vol, 3),
                "invol_lift": round(invol - 0.25, 3),
                "vol_lift":   round(vol - 0.25, 3),
            })

    res = pd.DataFrame(rows).sort_values("invol-vol", ascending=False)
    print("\n=== INTERACTION FEATURE LIFT TABLE ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_engineering.py <data_root>")
        print("  data_root should contain preprocessed/train/ and preprocessed/test/")
        sys.exit(1)

    root = Path(sys.argv[1])
    train_df = build_features(
        users_path       = root / "preprocessed/train/train_users.csv",
        props_path       = root / "preprocessed/train/train_users_properties.csv",
        purchases_path   = root / "preprocessed/train/train_users_purchases.csv",
        txn_path         = root / "preprocessed/train/train_users_transaction_attempts.csv",
        quizzes_path     = root / "preprocessed/train/train_users_quizzes.csv",
        generations_path = root / "preprocessed/train/train_users_generations.csv",
        is_train=True,
    )
    print_interaction_lift_table(train_df)
    train_df.to_parquet(root / "preprocessed/train_features.parquet", index=False)
    print("Saved → preprocessed/train_features.parquet")