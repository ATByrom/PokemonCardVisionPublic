import os
import csv
from pathlib import Path

RAW_DIR = Path("data/pokemon_photos/raw")
LABELS_CSV = Path("data/pokemon_photos/labels.csv")

# 👇 change these to whatever names you actually want
CLASS_KEYS = {
    "r": "rayquaza_v_100_159",   # Rayquaza V
    "g": "other_card",           # second card (rename later)
    "s": "skip",                 # skip / background / bad photo
    "q": "quit",
}

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load existing labels if any (so you can resume later)
    existing = {}
    if LABELS_CSV.exists():
        with LABELS_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["filename"]] = row["label"]
        print(f"[INFO] Loaded {len(existing)} existing labels from {LABELS_CSV}")

    images = sorted(
        [p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}],
        key=lambda p: p.name,
    )
    print(f"[INFO] Found {len(images)} images in {RAW_DIR}")

    # Open CSV for append (will keep existing rows)
    write_header = not LABELS_CSV.exists()
    f_csv = LABELS_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if write_header:
        writer.writerow(["filename", "label"])

    try:
        for img_path in images:
            rel_name = img_path.name

            # skip already-labeled
            if rel_name in existing:
                continue

            print("\n======================================")
            print(f"Image: {rel_name}")
            print("Open it in Explorer to peek if needed.")
            print("Labels:")
            for k, v in CLASS_KEYS.items():
                if k not in ("s", "q"):
                    print(f"  [{k}] -> {v}")
            print("  [s] -> skip image")
            print("  [q] -> quit labeling")

            while True:
                choice = input("Enter label key: ").strip().lower()
                if choice not in CLASS_KEYS:
                    print("  Invalid choice, try again.")
                    continue

                if choice == "q":
                    print("[INFO] Quitting labeling loop.")
                    f_csv.close()
                    return

                label = CLASS_KEYS[choice]
                if label == "skip":
                    print("  -> Skipping image.")
                else:
                    print(f"  -> Labelled as '{label}'")
                    writer.writerow([rel_name, label])
                    f_csv.flush()
                break

    finally:
        f_csv.close()
        print(f"[INFO] Saved labels to {LABELS_CSV}")


if __name__ == "__main__":
    main()