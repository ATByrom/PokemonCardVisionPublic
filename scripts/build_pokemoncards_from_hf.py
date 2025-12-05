import os
import csv
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Raw CSV URLs from HuggingFace
CARDS_URL = "https://huggingface.co/datasets/tooni/pokemoncards/resolve/main/cards.csv"
SETS_URL = "https://huggingface.co/datasets/tooni/pokemoncards/resolve/main/sets.csv"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pokemon_tcg_hf"
IMG_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "cards.csv"

IMG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_url(url: str) -> str:
    # strip whitespace and any stray quotes
    return url.strip().strip('"').strip("'")


def download_image(url: str, dest_path: Path, retries: int = 2, timeout: int = 15) -> bool:
    if dest_path.exists():
        return True

    # sanitize url
    url = clean_url(url)
    if not url.startswith("http"):
        print(f"[WARN] Bad URL '{url}'")
        return False

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                dest_path.write_bytes(resp.content)
                return True
            else:
                print(f"[WARN] {url} -> HTTP {resp.status_code}")
        except Exception as e:
            print(f"[WARN] Error fetching {url}: {e}")
        # small backoff, but not huge
        time.sleep(0.5 * attempt)

    return False


def main():
    print(">>> Downloading cards.csv from HuggingFace ...")
    cards_df = pd.read_csv(CARDS_URL)
    print(f"  Loaded {len(cards_df)} card rows")

    print(">>> Downloading sets.csv from HuggingFace ...")
    sets_df = pd.read_csv(SETS_URL)
    print(f"  Loaded {len(sets_df)} set rows")

    # cards_df has 'id' like 'sv1-1', 'xy1-4', etc.
    if "id" not in cards_df.columns:
        raise KeyError("cards.csv does not have an 'id' column as expected")

    # derive set_id from id prefix before '-'
    cards_df["set_id"] = cards_df["id"].astype(str).str.split("-").str[0]

    if "id" not in sets_df.columns or "name" not in sets_df.columns or "releaseDate" not in sets_df.columns:
        raise KeyError("sets.csv missing required columns: 'id', 'name', 'releaseDate'")

    # join cards with sets on set_id
    merged = cards_df.merge(
        sets_df[["id", "name", "releaseDate"]],
        left_on="set_id",
        right_on="id",
        how="left",
        suffixes=("", "_set"),
    )

    merged["set_name"] = merged["name_set"]
    merged["set_release_date"] = merged["releaseDate"]

    # convert release date to datetime and filter to ~last 10 years
    merged["set_release_date_dt"] = pd.to_datetime(
        merged["set_release_date"], errors="coerce"
    )
    cutoff = pd.Timestamp("2015-01-01")
    merged_recent = merged[merged["set_release_date_dt"] >= cutoff].copy()

    print(
        f">>> Cards with release_date >= {cutoff.date()}: "
        f"{len(merged_recent)}/{len(merged)}"
    )

    # expected columns in cards.csv:
    # small_image_source, large_image_source, number, etc.
    cols = [
        "id",
        "name",
        "set_id",
        "set_name",
        "set_release_date",
        "number",
        "small_image_source",
        "large_image_source",
    ]

    for c in cols:
        if c not in merged_recent.columns:
            print(f"[WARN] Expected column {c} missing from merged DataFrame")

    merged_recent = merged_recent[cols].copy()

    rows = []

    print(">>> Downloading images ...")
    for _, row in tqdm(merged_recent.iterrows(), total=len(merged_recent), desc="Cards"):
        card_id = row["id"]
        name = row["name"]
        set_id = row["set_id"]
        set_name = row["set_name"]
        set_release_date = row["set_release_date"]
        number = row["number"]
        small = row["small_image_source"]
        large = row["large_image_source"]

        url = None
        if isinstance(large, str) and large.strip():
            url = large
        elif isinstance(small, str) and small.strip():
            url = small

        if not isinstance(url, str) or not url.strip():
            continue

        # use card_id as filename to avoid collisions
        url_clean = clean_url(url)
        ext = os.path.splitext(url_clean)[1] or ".jpg"
        local_name = f"{card_id}{ext}"
        dest = IMG_DIR / local_name

        ok = download_image(url_clean, dest)
        if not ok:
            continue

        rows.append(
            {
                "id": card_id,
                "name": name,
                "set_id": set_id,
                "set_name": set_name,
                "set_release_date": set_release_date,
                "number": number,
                "image_small": small,
                "image_large": large,
                "local_path": str(dest.relative_to(ROOT)),
            }
        )

    if not rows:
        print("[WARN] No cards processed successfully; nothing to write.")
        return

    fieldnames = [
        "id",
        "name",
        "set_id",
        "set_name",
        "set_release_date",
        "number",
        "image_small",
        "image_large",
        "local_path",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f">>> Wrote {len(rows)} cards to {CSV_PATH}")
    print(f">>> Images saved under {IMG_DIR}")


if __name__ == "__main__":
    main()