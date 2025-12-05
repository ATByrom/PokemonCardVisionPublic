import csv
from pathlib import Path

import pandas as pd
from pillow_heif import register_heif_opener
from PIL import Image

# Enable HEIC/HEIF support
register_heif_opener()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "pokemon_photos" / "raw"
LABELS_CSV = BASE_DIR / "data" / "pokemon_photos" / "labels.csv"

# Map single-character keys to label strings
KEY_TO_LABEL = {
    "r": "rayquaza_v_100_159",
    "k": "kyogre_v_037_159",
    "h": "honchkrow_v_088_172",
    "z": "zamazenta_v_098_159",
    "s": "skip",  # will be dropped before training
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def load_existing_labels():
    """
    Load existing labels.csv if it exists.
    Returns a DataFrame and a set of already-labeled filenames.
    """
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        if "filename" not in df.columns or "label" not in df.columns:
            raise SystemExit(f"{LABELS_CSV} does not have 'filename' and 'label' columns")
        labeled = set(df["filename"].astype(str).tolist())
        print(f"[INFO] Loaded {len(df)} existing labels from {LABELS_CSV}")
        return df, labeled
    else:
        print(f"[INFO] No existing {LABELS_CSV}, starting fresh.")
        return pd.DataFrame(columns=["filename", "label"]), set()


def list_images():
    if not RAW_DIR.exists():
        raise SystemExit(f"Raw image folder not found: {RAW_DIR.resolve()}")

    images = sorted(
        [p for p in RAW_DIR.iterdir() if p.suffix.lower() in VALID_EXTS]
    )
    print(f"[INFO] Found {len(images)} images in {RAW_DIR}")
    return images


def main():
    df, labeled = load_existing_labels()
    images = list_images()

    # We will append new rows to this list and then concat once at the end
    new_rows = []

    print(
        "\n[INFO] Labeling controls:\n"
        "  r = rayquaza_v_100_159\n"
        "  k = kyogre_v_037_159\n"
        "  h = honchkrow_v_088_172\n"
        "  z = zamazenta_v_098_159\n"
        "  s = skip (do not use for training)\n"
        "  q = quit\n"
    )

    for img_path in images:
        fname = img_path.name

        # Skip if already labeled
        if fname in labeled:
            continue

        print(f"\n--- {fname} ---")
        print(f"Full path: {img_path}")
        # If you want to auto-open the image viewer you could add:
        # Image.open(img_path).show()

        while True:
            key = input("Enter label [r/k/h/z/s] or q to quit: ").strip().lower()

            if key == "q":
                print("[INFO] Quitting labeling loop.")
                # Save any new rows before exiting
                if new_rows:
                    df_new = pd.DataFrame(new_rows, columns=["filename", "label"])
                    df_out = pd.concat([df, df_new], ignore_index=True)
                    df_out.to_csv(LABELS_CSV, index=False)
                    print(f"[INFO] Saved updated labels to {LABELS_CSV}")
                return

            if key not in KEY_TO_LABEL:
                print("Invalid key. Use r/k/h/z/s or q.")
                continue

            label = KEY_TO_LABEL[key]
            new_rows.append((fname, label))
            print(f"[INFO] Labeled {fname} as '{label}'")
            break

    # Finished all images
    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=["filename", "label"])
        df_out = pd.concat([df, df_new], ignore_index=True)
        df_out.to_csv(LABELS_CSV, index=False)
        print(f"\n[INFO] Saved updated labels to {LABELS_CSV}")
    else:
        print("\n[INFO] No new labels added; everything was already labeled.")


if __name__ == "__main__":
    main()