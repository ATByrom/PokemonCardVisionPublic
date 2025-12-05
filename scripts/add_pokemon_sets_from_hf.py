import csv
import time
from pathlib import Path
from typing import List, Dict

import requests
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------
# CONFIG – EDIT THIS PART WHEN YOU WANT TO ADD NEW SETS
# ---------------------------------------------------------------------
# Put the set_id strings here that you want to import COMPLETELY.
# You’ll get set_id values from sets.csv (explained below).
#
# For your 3 Gyarados cards, you’ll likely use something like:
#   - Team Rocket                      -> e.g. "base5" (CHECK sets.csv)
#   - Promo Card Pack 25th Anniversary -> e.g. "s8a-P" (CHECK sets.csv)
#   - Pokémon TCG Classic Blastoise    -> e.g. "clb"   (CHECK sets.csv)
#
TARGET_SET_IDS: List[str] = [
    # "base5",
    # "s8a-P",
    # "clb",
]

# Where your main dataset lives
ROOT = Path(__file__).resolve().parents[1]   # repo root
DATA_DIR = ROOT / "data" / "pokemon_tcg_hf"
IMG_DIR = DATA_DIR / "images"
MASTER_CSV = DATA_DIR / "cards.csv"

# HuggingFace dataset name / files
HF_REPO = "tooni/pokemoncards"
HF_CARDS_FILENAME = "cards.csv"
HF_SETS_FILENAME = "sets.csv"

REQUEST_TIMEOUT = 20
REQUEST_RETRIES = 3
REQUEST_SLEEP = 2  # seconds between retries


# ---------------------------------------------------------------------
# Helper: safe image download
# ---------------------------------------------------------------------
def download_image(url: str, dest: Path) -> bool:
    url = url.strip()
    # If URL is quoted like `"https://..."`, strip inner quotes
    if url.startswith('"') and url.endswith('"'):
        url = url[1:-1].strip()

    if not url:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        # Already downloaded
        return True

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            with dest.open("wb") as f:
                f.write(resp.content)
            return True
        except Exception as e:
            print(
                f"[WARN] Error fetching {url}: {e} "
                f"(attempt {attempt}/{REQUEST_RETRIES})"
            )
            time.sleep(REQUEST_SLEEP * attempt)

    return False


# ---------------------------------------------------------------------
# Load HF CSVs
# ---------------------------------------------------------------------
def load_hf_cards() -> List[Dict]:
    cards_path = hf_hub_download(HF_REPO, HF_CARDS_FILENAME)
    rows: List[Dict] = []
    with open(cards_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[INFO] Loaded {len(rows)} rows from HF cards.csv")
    return rows


def load_hf_sets() -> Dict[str, Dict]:
    sets_path = hf_hub_download(HF_REPO, HF_SETS_FILENAME)
    sets: Dict[str, Dict] = {}
    with open(sets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("set_id") or row.get("id")
            if sid:
                sets[sid] = row
    print(f"[INFO] Loaded {len(sets)} rows from HF sets.csv")
    return sets


# ---------------------------------------------------------------------
# Load existing master CSV (your current dataset)
# ---------------------------------------------------------------------
def load_master() -> (List[Dict], Dict[str, Dict]):
    rows: List[Dict] = []
    existing_by_id: Dict[str, Dict] = {}

    if not MASTER_CSV.exists():
        print(f"[WARN] MASTER CSV not found at {MASTER_CSV}, treating as empty.")
        return rows, existing_by_id

    with MASTER_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            cid = row.get("id")
            if cid:
                existing_by_id[cid] = row

    print(
        f"[INFO] Loaded existing master CSV: {len(rows)} rows "
        f"({len(existing_by_id)} unique card IDs)"
    )
    return rows, existing_by_id


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if not TARGET_SET_IDS:
        print(
            "ERROR: TARGET_SET_IDS is empty.\n"
            "Edit add_pokemon_sets_from_hf.py and add one or more set_id strings "
            "to TARGET_SET_IDS at the top of the file."
        )
        return

    print(f">>> Adding full sets for set_ids: {', '.join(TARGET_SET_IDS)}")

    hf_cards = load_hf_cards()
    hf_sets = load_hf_sets()

    master_rows, existing_by_id = load_master()

    # Build a quick lookup of HF sets for display
    for sid in TARGET_SET_IDS:
        srow = hf_sets.get(sid)
        if srow:
            print(
                f"[INFO] Target set {sid}: "
                f"{srow.get('set_name') or srow.get('name')} "
                f"(release: {srow.get('releaseDate') or srow.get('release_date')})"
            )
        else:
            print(f"[WARN] set_id {sid} not found in HF sets.csv")

    # Filter HF cards for target sets
    new_cards: List[Dict] = []
    for row in hf_cards:
        set_id = row.get("set_id") or row.get("set")
        if not set_id:
            continue

        if set_id not in TARGET_SET_IDS:
            continue

        cid = row.get("id")
        if not cid:
            continue

        if cid in existing_by_id:
            # Already in master dataset
            continue

        new_cards.append(row)

    if not new_cards:
        print("[INFO] No new cards found for the requested sets.")
        return

    print(f"[INFO] Found {len(new_cards)} new cards to add.")

    # Download images + build normalized rows to append
    appended_rows: List[Dict] = []

    for row in new_cards:
        cid = row["id"]
        name = row.get("name", "")
        set_id = row.get("set_id") or row.get("set") or ""
        set_info = hf_sets.get(set_id, {})
        set_name = (
            set_info.get("set_name")
            or set_info.get("name")
            or row.get("set_name")
            or ""
        )
        release = (
            set_info.get("releaseDate")
            or set_info.get("release_date")
            or row.get("set_release_date")
            or ""
        )
        number = row.get("number", "")

        image_small = (
            row.get("small_image_source")
            or row.get("image_small")
            or ""
        )
        image_large = (
            row.get("large_image_source")
            or row.get("image_large")
            or ""
        )

        # Decide filename based on card id
        ext = ".png"
        lower_url = (image_large or image_small).lower()
        if ".jpg" in lower_url or ".jpeg" in lower_url:
            ext = ".jpg"

        out_path = IMG_DIR / f"{cid}{ext}"
        local_rel = out_path.relative_to(ROOT)

        ok = False
        if image_large:
            ok = download_image(image_large, out_path)
        elif image_small:
            ok = download_image(image_small, out_path)

        if not ok:
            print(f"[WARN] Skipping card {cid} ({name}) – could not download image.")
            continue

        appended_rows.append(
            {
                "id": cid,
                "name": name,
                "set_id": set_id,
                "set_name": set_name,
                "set_release_date": release,
                "number": number,
                "image_small": image_small,
                "image_large": image_large,
                "local_path": str(local_rel),
            }
        )

    if not appended_rows:
        print("[WARN] No new rows appended (all failed image download?).")
        return

    print(f"[INFO] Successfully prepared {len(appended_rows)} appended rows.")

    # Merge and rewrite master CSV
    all_rows = master_rows + appended_rows

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

    MASTER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(
        f"[DONE] Master CSV updated: {len(all_rows)} total rows "
        f"(added {len(appended_rows)} new cards)."
    )


if __name__ == "__main__":
    main()