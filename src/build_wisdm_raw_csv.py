import os
import csv
import argparse

# WISDM activity codes -> names (matches dataset docs)
ACT = {
    "A": "Walking", "B": "Jogging", "C": "Upstairs", "D": "Downstairs",
    "E": "Sitting", "F": "Standing", "G": "Typing", "H": "BrushingTeeth",
    "I": "EatingSoup", "J": "EatingChips", "K": "EatingPasta", "L": "Drinking",
    "M": "EatingSandwich", "O": "Kicking", "P": "PlayingCatch", "Q": "Dribbling",
    "R": "Writing", "S": "Clapping"  # some releases list S=FoldingClothes; either is fine
}

def parse_line(line: str):
    """
    Raw WISDM line (phone/*/data_*.txt):
      user, activity_code, timestamp, ax, ay, az;
    Returns: (user:str, activity:str, t:float|None, ax:float, ay:float, az:float) or None if bad.
    """
    line = line.strip()
    if not line:
        return None
    if line.endswith(";"):
        line = line[:-1]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return None

    user, code, ts, x, y, z = parts[:6]

    try:
        ax, ay, az = float(x), float(y), float(z)
    except ValueError:
        return None

    try:
        t = float(ts)
    except ValueError:
        t = None

    return user, ACT.get(code, code), t, ax, ay, az


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to wisdm-dataset/raw")
    ap.add_argument("--out", default="data/WISDM_raw.csv")
    ap.add_argument("--sensor", choices=["accel", "gyro"], default="accel",
                    help="Which phone sensor subfolder to read")
    ap.add_argument("--rate", type=float, default=20.0,
                    help="Hz, only used to synthesize timestamps if missing")
    # Defaults tuned for a small (KB-size) CSV that still supports 10-fold CV
    ap.add_argument("--max_users", type=int, default=3,
                    help="Keep the first N distinct users encountered")
    ap.add_argument("--every_k", type=int, default=100,
                    help="Temporal downsampling: keep every k-th row per file (1 = no downsample)")
    ap.add_argument("--max_rows", type=int, default=0,
                    help="If >0, stop after writing N rows total (hard cap for file size)")
    args = ap.parse_args()

    # Using PHONE sensor to keep it simple
    src_dir = os.path.join(args.root, "phone", args.sensor)
    if not os.path.isdir(src_dir):
        raise SystemExit(f"Not found: {src_dir}")

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Discover files (e.g., data_1600_accel_phone.txt)
    file_names = [n for n in sorted(os.listdir(src_dir)) if n.startswith("data_")]
    def user_from_name(n: str):
        # expected pattern: data_<user>_<sensor>_phone.txt
        parts = n.split("_")
        return parts[1] if len(parts) >= 2 else None

    # Choose first N distinct users (by the file order)
    chosen_users = []
    for name in file_names:
        u = user_from_name(name)
        if not u:
            continue
        if u not in chosen_users:
            chosen_users.append(u)
            if len(chosen_users) >= args.max_users:
                break

    if not chosen_users:
        raise SystemExit("No users found in the provided directory.")

    print(f"Including users: {', '.join(chosen_users)} (total {len(chosen_users)})")

    rows = []
    for name in file_names:
        u_in_name = user_from_name(name)
        if u_in_name not in chosen_users:
            continue

        path = os.path.join(src_dir, name)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            fallback_idx = 0
            keep_idx = 0
            for line in f:
                # Downsample: keep every k-th row
                if args.every_k > 1:
                    if (keep_idx % args.every_k) != 0:
                        keep_idx += 1
                        continue
                    keep_idx += 1

                rec = parse_line(line)
                if not rec:
                    continue
                user, act, t, ax, ay, az = rec

                # Only accept rows where the inline user matches the file's user
                if user != u_in_name:
                    continue

                # Synthesize timestamp if missing
                if t is None:
                    t = fallback_idx / args.rate
                    fallback_idx += 1

                rows.append((user, act, t, ax, ay, az))

                # Hard cap for file size control (optional)
                if args.max_rows and len(rows) >= args.max_rows:
                    break

        if args.max_rows and len(rows) >= args.max_rows:
            break

    # Sort by (user, timestamp) for better viewing
    rows.sort(key=lambda r: (r[0], r[2]))

    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user", "activity", "timestamp", "ax", "ay", "az"])
        w.writerows(rows)

    print(f"Wrote {len(rows):,} rows to {args.out}")


if __name__ == "__main__":
    main()
