import csv
import os
from datetime import datetime
from collections import defaultdict

PLACEHOLDER = None


def parse_dt(s):
    if not s or not s.strip():
        return None
    try:
        s = s.strip()
        if "+" in s:
            s = s[: s.rfind("+")]
        elif s.endswith("Z"):
            s = s[:-1]
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def classify(gen_type):
    if not gen_type:
        return "overall"
    g = gen_type.lower()
    if "video" in g:
        return "video"
    if "image" in g:
        return "image"
    return "overall"


def process_file(input_path, output_path):
    rows_by_user = defaultdict(list)
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_user[row["user_id"]].append(row)

    # ── Phase 1: completion duration (completed_at - created_at) ──
    dur_overall_valid = []
    dur_image_valid = []
    dur_video_valid = []

    user_dur = {}
    for uid, rows in rows_by_user.items():
        buckets = {"overall": [], "image": [], "video": []}
        placeholders = {"overall": 0, "image": 0, "video": 0}

        for r in rows:
            cat = classify(r.get("generation_type", ""))
            created = parse_dt(r.get("created_at", ""))
            completed = parse_dt(r.get("completed_at", ""))

            if created and completed:
                diff = (completed - created).total_seconds() / 60.0
                buckets["overall"].append(diff)
                if cat in ("image", "video"):
                    buckets[cat].append(diff)
            else:
                placeholders["overall"] += 1
                if cat in ("image", "video"):
                    placeholders[cat] += 1

        avgs = {}
        for key in ("overall", "image", "video"):
            if buckets[key]:
                avgs[key] = sum(buckets[key]) / len(buckets[key])
            elif placeholders[key] > 0:
                avgs[key] = PLACEHOLDER
            else:
                avgs[key] = 0.0

        dur_overall_valid.extend(buckets["overall"])
        dur_image_valid.extend(buckets["image"])
        dur_video_valid.extend(buckets["video"])

        user_dur[uid] = {"avgs": avgs, "placeholders": placeholders}

    global_dur_overall = sum(dur_overall_valid) / len(dur_overall_valid) if dur_overall_valid else 0.0
    global_dur_image = sum(dur_image_valid) / len(dur_image_valid) if dur_image_valid else 0.0
    global_dur_video = sum(dur_video_valid) / len(dur_video_valid) if dur_video_valid else 0.0

    for uid in user_dur:
        for key, gval in [("overall", global_dur_overall), ("image", global_dur_image), ("video", global_dur_video)]:
            if user_dur[uid]["avgs"][key] is PLACEHOLDER:
                user_dur[uid]["avgs"][key] = gval

    # ── Phase 2: focus (avg gap between sequential created_at) ──
    focus_overall_valid = []
    focus_image_valid = []
    focus_video_valid = []

    user_focus = {}
    for uid, rows in rows_by_user.items():
        ts_overall = []
        ts_image = []
        ts_video = []

        for r in rows:
            cat = classify(r.get("generation_type", ""))
            created = parse_dt(r.get("created_at", ""))
            if created:
                ts_overall.append(created)
                if cat == "image":
                    ts_image.append(created)
                elif cat == "video":
                    ts_video.append(created)

        def avg_gaps(timestamps):
            if len(timestamps) < 2:
                return PLACEHOLDER if len(timestamps) <= 1 else 0.0
            timestamps.sort()
            gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0 for i in range(len(timestamps) - 1)]
            return sum(gaps) / len(gaps)

        avgs = {}
        placeholders = {"overall": 0, "image": 0, "video": 0}
        for key, ts_list, valid_list in [
            ("overall", ts_overall, focus_overall_valid),
            ("image", ts_image, focus_image_valid),
            ("video", ts_video, focus_video_valid),
        ]:
            val = avg_gaps(ts_list)
            if val is PLACEHOLDER:
                placeholders[key] = 1
            else:
                valid_list.append(val)
            avgs[key] = val

        user_focus[uid] = {"avgs": avgs, "placeholders": placeholders}

    global_focus_overall = sum(focus_overall_valid) / len(focus_overall_valid) if focus_overall_valid else 0.0
    global_focus_image = sum(focus_image_valid) / len(focus_image_valid) if focus_image_valid else 0.0
    global_focus_video = sum(focus_video_valid) / len(focus_video_valid) if focus_video_valid else 0.0

    for uid in user_focus:
        for key, gval in [("overall", global_focus_overall), ("image", global_focus_image), ("video", global_focus_video)]:
            if user_focus[uid]["avgs"][key] is PLACEHOLDER:
                user_focus[uid]["avgs"][key] = gval

    # ── Write output ──
    all_uids = sorted(rows_by_user.keys())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "user_id",
            "avg_completion_duration_min_overall",
            "avg_completion_duration_min_image",
            "avg_completion_duration_min_video",
            "avg_focus_gap_min_overall",
            "avg_focus_gap_min_image",
            "avg_focus_gap_min_video",
        ])
        for uid in all_uids:
            writer.writerow([
                uid,
                round(user_dur[uid]["avgs"]["overall"], 4),
                round(user_dur[uid]["avgs"]["image"], 4),
                round(user_dur[uid]["avgs"]["video"], 4),
                round(user_focus[uid]["avgs"]["overall"], 4),
                round(user_focus[uid]["avgs"]["image"], 4),
                round(user_focus[uid]["avgs"]["video"], 4),
            ])

    print(f"Written {len(all_uids)} users to {output_path}")
    print(f"  Global completion duration (min) -> overall={global_dur_overall:.4f}, image={global_dur_image:.4f}, video={global_dur_video:.4f}")
    print(f"  Global focus gap (min)           -> overall={global_focus_overall:.4f}, image={global_focus_image:.4f}, video={global_focus_video:.4f}")


process_file(
    "data/preprocessed/train/train_users_generations.csv",
    "data/durations_train.csv",
)

process_file(
    "data/preprocessed/test/test_users_generations.csv",
    "data/durations_test.csv",
)