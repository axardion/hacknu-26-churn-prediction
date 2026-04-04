#!/usr/bin/env python3
"""
Preprocess train/test CSVs: quiz column normalization, role frequency filters,
drop flow_type, replace empty strings with 'skipped' everywhere, and drop train
rows whose country_code (from properties) never appears in the test set.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _strip(s: str | None) -> str:
    if s is None:
        return ""
    return str(s).strip()


def empty_to_skipped(val: str) -> str:
    return "skipped" if _strip(val) == "" else val


# --- Quiz transforms ---

SOLO_1 = frozenset({"solo", "1"})
LARGE_TIER = frozenset({"large", "5000+", "2001-5000"})
MID_TIER = frozenset({"midsize", "51-500", "501-2000"})
SMALL_TIER = frozenset({"small", "11-50", "2-10"})
TEAM_AS_IS = frozenset({"growing", "enterprise"})


def map_team_size(raw: str) -> str:
    v = _strip(raw)
    if not v:
        return ""
    if v in SOLO_1:
        return "solo_1"
    if v in LARGE_TIER:
        return "large_tier"
    if v in MID_TIER:
        return "midsize_tier"
    if v in SMALL_TIER:
        return "small_tier"
    if v in TEAM_AS_IS:
        return v
    return v


ALLOWED_EXPERIENCE = frozenset({"beginner", "intermediate", "advanced", "expert"})


def map_experience(raw: str) -> str:
    v = _strip(raw)
    if not v:
        return ""
    return v if v in ALLOWED_EXPERIENCE else "skipped"


FRUSTRATION_CANON = {
    "high-cost": "high-cost",
    "high cost of top models": "high-cost",
    "hard-prompt": "hard-prompt",
    "hard to prompt": "hard-prompt",
    "confusing": "confusing",
    "ai is confusing to me": "confusing",
    "limited": "limited",
    "limited generations": "limited",
    "inconsistent": "inconsistent",
    "inconsistent results": "inconsistent",
    "other": "other",
}


def map_frustration(raw: str) -> str:
    v = _strip(raw)
    if not v:
        return ""
    key = v.lower()
    return FRUSTRATION_CANON.get(key, v)


# first_feature merge groups (lowercase key -> canonical label)
_FIRST_FEATURE_ALIASES: dict[str, str] = {}
for _label, _variants in [
    (
        "video_generation",
        (
            "video generations",
            "video-creation",
            "video generation",
        ),
    ),
    (
        "commercial_ads",
        (
            "commercial & ad videos",
            "product-placement",
        ),
    ),
    (
        "avatars",
        (
            "realistic ai avatars",
            "realistic avatars & ai twins",
            "consistent-character",
            "lipsync & talking avatars",
            "talking-avatars",
        ),
    ),
    (
        "image_editing",
        (
            "image editing & inpaint",
            "edit-image",
        ),
    ),
    (
        "image_creation",
        (
            "image-creation",
            "image generation",
        ),
    ),
    (
        "upscale",
        (
            "upscale",
        ),
    ),
]:
    for _v in _variants:
        _FIRST_FEATURE_ALIASES[_v] = _label


def map_first_feature(raw: str) -> str:
    v = _strip(raw)
    if not v:
        return ""
    key = v.lower()
    return _FIRST_FEATURE_ALIASES.get(key, v)


def map_source(raw: str) -> str:
    v = _strip(raw)
    if not v:
        return ""
    if v.lower() == "rofl":
        return "other"
    return v


def map_usage_plan(raw: str) -> str:
    v = _strip(raw)
    if not v:
        return ""
    if v == "team":
        return "skipped"
    return v


def map_role(role: str, allowed: frozenset[str]) -> str:
    v = _strip(role)
    if not v:
        return ""
    return v if v in allowed else "skipped"


def count_column(path: Path, col: str) -> Counter[str]:
    c: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        idx = header.index(col)
        for row in r:
            if len(row) <= idx:
                continue
            v = _strip(row[idx])
            c[v if v else ""] += 1
    return c


def collect_test_country_codes(test_properties: Path) -> frozenset[str]:
    """Country codes in test properties after the same empty -> skipped normalization."""
    codes: set[str] = set()
    with test_properties.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        i_cc = header.index("country_code")
        for row in r:
            raw = row[i_cc] if len(row) > i_cc else ""
            codes.add(empty_to_skipped(raw))
    return frozenset(codes)


def collect_allowed_train_user_ids(
    train_properties: Path, test_countries: frozenset[str]
) -> tuple[frozenset[str], int]:
    """Train user_ids whose normalized country_code appears in the test country set.

    Returns (allowed_ids, row_count) where row_count is data rows in properties.
    """
    allowed: set[str] = set()
    n_rows = 0
    with train_properties.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        i_uid = header.index("user_id")
        i_cc = header.index("country_code")
        for row in r:
            n_rows += 1
            uid = _strip(row[i_uid]) if len(row) > i_uid else ""
            raw_cc = row[i_cc] if len(row) > i_cc else ""
            cc = empty_to_skipped(raw_cc)
            if cc in test_countries:
                allowed.add(uid)
    return frozenset(allowed), n_rows


def collect_transaction_ids_for_users(
    purchases_path: Path, allowed_user_ids: frozenset[str]
) -> frozenset[str]:
    """Transaction IDs from purchase rows belonging to allowed users."""
    ids: set[str] = set()
    with purchases_path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        i_uid = header.index("user_id")
        i_txn = header.index("transaction_id")
        for row in r:
            if len(row) <= max(i_uid, i_txn):
                continue
            if _strip(row[i_uid]) in allowed_user_ids:
                ids.add(_strip(row[i_txn]))
    return frozenset(ids)


def build_role_allowlists(
    train_quiz: Path, test_quiz: Path
) -> tuple[frozenset[str], frozenset[str]]:
    tr = count_column(train_quiz, "role")
    te = count_column(test_quiz, "role")
    train_ok = frozenset(
        {k for k, n in tr.items() if k and n >= 750}
    )
    test_ok = frozenset({k for k, n in te.items() if k and n >= 4})
    return train_ok, test_ok


def transform_quiz_row(
    row: list[str],
    header: list[str],
    role_allow: frozenset[str],
) -> list[str]:
    hmap = {name: i for i, name in enumerate(header)}
    out = list(row)

    def get(name: str) -> str:
        i = hmap[name]
        return out[i] if i < len(out) else ""

    def set_(name: str, val: str) -> None:
        i = hmap[name]
        while len(out) <= i:
            out.append("")
        out[i] = val

    set_("source", map_source(get("source")))
    set_("team_size", map_team_size(get("team_size")))
    set_("experience", map_experience(get("experience")))
    set_("usage_plan", map_usage_plan(get("usage_plan")))
    set_("frustration", map_frustration(get("frustration")))
    set_("first_feature", map_first_feature(get("first_feature")))
    set_("role", map_role(get("role"), role_allow))

    # drop flow_type
    ft_i = hmap.get("flow_type")
    if ft_i is not None:
        out.pop(ft_i)
    return out


def transform_quiz_header(header: list[str]) -> list[str]:
    h = list(header)
    if "flow_type" in h:
        h.remove("flow_type")
    return h


def apply_empty_skipped_row(fields: list[str]) -> list[str]:
    return [empty_to_skipped(x) for x in fields]


def process_csv_quizzes(
    src: Path,
    dst: Path,
    role_allow: frozenset[str],
    *,
    allowed_user_ids: frozenset[str] | None = None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8") as fin, dst.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        header = next(reader)
        new_header = transform_quiz_header(header)
        writer = csv.writer(fout)
        writer.writerow(new_header)

        hmap = {name: i for i, name in enumerate(header)}
        i_uid = hmap["user_id"]
        out_i = 0
        for row in reader:
            max_i = max(hmap.values()) if hmap else -1
            while len(row) <= max_i:
                row.append("")
            uid = _strip(row[i_uid])
            if allowed_user_ids is not None and uid not in allowed_user_ids:
                continue
            transformed = transform_quiz_row(row, header, role_allow)
            row_out = apply_empty_skipped_row(transformed)
            if row_out:
                row_out = list(row_out)
                row_out[0] = str(out_i)
            out_i += 1
            writer.writerow(row_out)


def _process_csv_generic_impl(
    src: Path,
    dst: Path,
    *,
    allowed_user_ids: frozenset[str] | None = None,
    renumber_index: bool = False,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8") as fin, dst.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        header = next(reader)
        writer = csv.writer(fout)
        try:
            i_uid = header.index("user_id")
        except ValueError:
            i_uid = None
        out_i = 0
        writer.writerow(header)
        for row in reader:
            while len(row) < len(header):
                row.append("")
            if allowed_user_ids is not None and i_uid is not None:
                uid = _strip(row[i_uid])
                if uid not in allowed_user_ids:
                    continue
            row = apply_empty_skipped_row(row)
            if renumber_index and row:
                row = list(row)
                row[0] = str(out_i)
                out_i += 1
            writer.writerow(row)


def process_csv_generic(
    src: Path,
    dst: Path,
    *,
    allowed_user_ids: frozenset[str] | None = None,
    renumber_index: bool = False,
) -> None:
    return _process_csv_generic_impl(
        src, dst, allowed_user_ids=allowed_user_ids, renumber_index=renumber_index
    )


def process_csv_transaction_attempts(
    src: Path,
    dst: Path,
    allowed_transaction_ids: frozenset[str],
    *,
    renumber_index: bool = True,
) -> None:
    """Keep rows whose transaction_id is in allowed_transaction_ids."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline="", encoding="utf-8") as fin, dst.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.reader(fin)
        header = next(reader)
        i_txn = header.index("transaction_id")
        writer = csv.writer(fout)
        writer.writerow(header)
        out_i = 0
        for row in reader:
            while len(row) <= i_txn:
                row.append("")
            tid = _strip(row[i_txn])
            if tid not in allowed_transaction_ids:
                continue
            row = apply_empty_skipped_row(row)
            if renumber_index and row:
                row = list(row)
                row[0] = str(out_i)
                out_i += 1
            writer.writerow(row)


def process_large_csv(
    src: Path,
    dst: Path,
    *,
    allowed_user_ids: frozenset[str] | None = None,
) -> None:
    """Stream copy with empty -> skipped; optional filter on user_id."""
    _process_csv_generic_impl(
        src,
        dst,
        allowed_user_ids=allowed_user_ids,
        renumber_index=allowed_user_ids is not None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess hack-nu-26 train/test CSVs.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Root with train/ and test/ subfolders",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed"),
        help="Output root (mirrors train/ and test/)",
    )
    p.add_argument(
        "--skip-generations",
        action="store_true",
        help="Do not copy train_users_generations / test_users_generations (large files).",
    )
    args = p.parse_args()
    root: Path = args.input_dir
    out_root: Path = args.output_dir

    train_quiz = root / "train" / "train_users_quizzes.csv"
    test_quiz = root / "test" / "test_users_quizzes.csv"
    train_props = root / "train" / "train_users_properties.csv"
    test_props = root / "test" / "test_users_properties.csv"
    train_purchases_src = root / "train" / "train_users_purchases.csv"

    train_roles, test_roles = build_role_allowlists(train_quiz, test_quiz)

    test_country_codes = collect_test_country_codes(test_props)
    allowed_train_users, n_train_props_rows = collect_allowed_train_user_ids(
        train_props, test_country_codes
    )
    train_txn_ids = collect_transaction_ids_for_users(
        train_purchases_src, allowed_train_users
    )

    files_train = [
        "train_users.csv",
        "train_users_properties.csv",
        "train_users_quizzes.csv",
        "train_users_purchases.csv",
        "train_users_transaction_attempts.csv",
        "train_users_generations.csv",
    ]
    files_test = [
        "test_users.csv",
        "test_users_properties.csv",
        "test_users_quizzes.csv",
        "test_users_purchases.csv",
        "test_users_transaction_attempts.csv",
        "test_users_generations.csv",
    ]

    process_csv_quizzes(
        train_quiz,
        out_root / "train" / "train_users_quizzes.csv",
        train_roles,
        allowed_user_ids=allowed_train_users,
    )
    process_csv_quizzes(
        test_quiz,
        out_root / "test" / "test_users_quizzes.csv",
        test_roles,
        allowed_user_ids=None,
    )

    # --- all other files: empty -> skipped only; train filtered by country ---
    for name in files_train:
        if name == "train_users_quizzes.csv":
            continue
        src = root / "train" / name
        dst = out_root / "train" / name
        if name == "train_users_generations.csv":
            if args.skip_generations:
                continue
            process_large_csv(src, dst, allowed_user_ids=allowed_train_users)
        elif name == "train_users_transaction_attempts.csv":
            process_csv_transaction_attempts(
                src, dst, train_txn_ids, renumber_index=True
            )
        else:
            process_csv_generic(
                src,
                dst,
                allowed_user_ids=allowed_train_users,
                renumber_index=True,
            )

    for name in files_test:
        if name == "test_users_quizzes.csv":
            continue
        src = root / "test" / name
        dst = out_root / "test" / name
        if name == "test_users_generations.csv":
            if args.skip_generations:
                continue
            process_large_csv(src, dst)
        else:
            process_csv_generic(src, dst)

    print("Wrote preprocessed files to", out_root.resolve())
    print("Test country_code values (distinct, after empty->skipped):", len(test_country_codes))
    print(
        "Train users kept (country in test set):",
        len(allowed_train_users),
        "/",
        n_train_props_rows,
    )
    print("Train role allowlist (>=750):", sorted(train_roles))
    print("Test role allowlist (>=4):", sorted(test_roles))


if __name__ == "__main__":
    main()
