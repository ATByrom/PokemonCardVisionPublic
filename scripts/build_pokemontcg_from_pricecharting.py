"""
build_pokemontcg_from_pricecharting.py

Use existing PriceCharting metadata (cards.csv) to query the
Pokémon TCG API one card at a time and download images + metadata.

Assumptions about cards.csv (from your PriceCharting scraper):
- Located at: data/web_raw/cards.csv
- Has at least columns: name, number
- Optionally: set_name (we'll store it for reference)

Pokémon TCG API docs:
    https://docs.pokemontcg.io/api-reference/cards/search-cards/

We will:
- For each row in cards.csv:
    - Build q = 'name:"<name>" number:"<number>"'
    - GET /v2/cards?q=...
    - Take the first match, download its image, store its metadata.
- Save:
    - Images to: data/pokemontcg_raw/images/<set_id>/<number>_<name>.jpg
    - Metadata to: data/pokemontcg_raw/images.csv
"""

import os
import csv
import time
import requests
from pathlib import Path
from urllib.parse import quote_plus

API_KEY = os.environ.get("POKEMONTCG_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing POKEMONTCG_API_KEY env var. Set it via:\n"
        '    setx POKEMONTCG_API_KEY "your_key_here"'
    )

BASE_URL = "https://api.pokemontcg.io/v2"
HEADERS = {"X-Api-Key": API_KEY}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PC_CSV = PROJECT_ROOT / "data" / "web_raw" / "cards.csv"

DATA_ROOT = PROJECT_ROOT / "data" / "pokemontcg_raw"
IMAGES_ROOT = DATA_ROOT / "images"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY = 0.5   # be extra gentle on the API
IMG_DELAY = 0.05


def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


def download_image(url: str, dest: Path) -> bool:
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        r = requests.get(url, stream=True, timeout=25)
        if r.status_code != 200:
            print(f"[WARN] Image GET {url} -> {r.status_code}")
            return False
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"[WARN] Image error {url}: {e}")
        return False


def search_card(name: str, number: str):
    """
    Search /v2/cards for a card by name + number.
    Returns first match's card dict or None.
    """
    # Build q string as documented
    # NOTE: We do not manually encode q; requests will handle it as a param.
    q = f'name:"{name}" number:"{number}"'

    params = {"q": q}
    url = f"{BASE_URL}/cards"

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=25)
    except Exception as e:
        print(f"[WARN] Search error for {name} #{number}: {e}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] Search {name} #{number} -> {resp.status_code}")
        return None

    try:
        body = resp.json()
    except Exception as e:
        print(f"[WARN] JSON error for {name} #{number}: {e}")
        return None

    data = body.get("data", [])
    if not data:
        print(f"[INFO] No API results for {name} #{number}")
        return None

    # Take the first match
    return data[0]


def main():
    if not PC_CSV.exists():
        raise FileNotFoundError(f"PriceCharting CSV not found at: {PC_CSV}")

    print(f">>> Using PriceCharting metadata from: {PC_CSV}")

    # Read all PriceCharting rows
    pc_rows = []
    with open(PC_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pc_rows.append(row)

    print(f"Loaded {len(pc_rows)} PriceCharting card entries")

    out_rows = []

    for i, row in enumerate(pc_rows, start=1):
        name = (row.get("name") or "").strip()
        number = (row.get("number") or "").strip()
        pc_set_name = (row.get("set_name") or row.get("set") or "").strip()

        if not name or not number:
            print(f"[WARN] Skipping row {i}: missing name/number")
            continue

        print(f"[{i}/{len(pc_rows)}] Searching API for {name} #{number} ...")
        card = search_card(name, number)
        time.sleep(REQUEST_DELAY)

        if not card:
            continue

        set_info = card.get("set") or {}
        images = card.get("images") or {}
        img_url = images.get("large") or images.get("small")
        if not img_url:
            print(f"[WARN] No image for {name} #{number} in API result")
            continue

        card_id = card.get("id", "")
        set_id = set_info.get("id", "")
        set_name = set_info.get("name", "")

        release_date = set_info.get("releaseDate", "")
        rarity = card.get("rarity", "")
        supertype = card.get("supertype", "")
        subtypes = ", ".join(card.get("subtypes", []) or [])
        artist = card.get("artist", "")

        # Use API set_id + card name/number for file path
        filename = slugify(f"{number}_{name}") + ".jpg"
        dest = IMAGES_ROOT / set_id / filename

        if not download_image(img_url, dest):
            continue

        out_rows.append(
            {
                "card_id": card_id,
                "set_id": set_id,
                "set_name": set_name,
                "pc_set_name": pc_set_name,
                "release_date": release_date,
                "number": number,
                "name": name,
                "rarity": rarity,
                "supertype": supertype,
                "subtypes": subtypes,
                "artist": artist,
                "image_url": img_url,
                "image_path": str(dest.as_posix()),
            }
        )

        time.sleep(IMG_DELAY)

    # Write images.csv (overwrite each time)
    csv_path = DATA_ROOT / "images.csv"
    if out_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

        print("\n>>> DONE")
        print(f"Total successful cards: {len(out_rows)}")
        print(f"Images dir: {IMAGES_ROOT}")
        print(f"Metadata CSV: {csv_path}")
    else:
        print("\n[WARN] No cards matched / no images downloaded; CSV not written.")


if __name__ == "__main__":
    main()