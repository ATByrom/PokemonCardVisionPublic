# scripts/update_labels_multi_class.py
from pathlib import Path
import pandas as pd

LABELS_CSV = Path("data/pokemon_photos/labels.csv")

# ===== EDIT THESE LISTS =====
# Put the exact filenames (no path) for each card here.
# Check data/pokemon_photos/raw in Explorer and copy the names.

KYOGRE_FILES = [
    # "IMG_5231.jpg",
    # "IMG_5232.HEIC",
]

HONCHKROW_FILES = [
    # "IMG_5229.jpg",
]

ZAMAZENTA_FILES = [
    # "IMG_5230.jpg",
]

# If you have any Rayquaza images that *aren't* already labeled correctly,
# you can list them here too (optional):
RAYQUAZA_FILES = [
    # "IMG_51xx.jpg",
]

LABEL_MAP = {}
LABEL_MAP.update({fn: "kyogre_v_037_159" for fn in KYOGRE_FILES})
LABEL_MAP.update({fn: "honchkrow_v_088_172" for fn in HONCHKROW_FILES})
LABEL_MAP.update({fn: "zamazenta_v_098_159" for fn in ZAMAZENTA_FILES})
LABEL_MAP.update({fn: "rayquaza_v_100_159" for fn in RAYQUAZA_FILES})


def main():
    if not LABELS_CSV.exists():
        raise SystemExit(f"labels.csv not found at {LABELS_CSV.resolve()}")

    df = pd.read_csv(LABELS_CSV)

    # For any filename in LABEL_MAP:
    # - if it exists in df -> update label
    # - else -> append a new row
    existing = set(df["filename"].tolist())
    rows_to_add = []

    for fn, label in LABEL_MAP.items():
        if fn in existing:
            df.loc[df["filename"] == fn, "label"] = label
        else:
            rows_to_add.append({"filename": fn, "label": label})

    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Optional: if you want to completely get rid of the generic other_card label:
    # df = df[df["label"] != "other_card"].reset_index(drop=True)

    df.to_csv(LABELS_CSV, index=False)
    print(f"Saved updated labels to {LABELS_CSV}")
    print("Class counts:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()