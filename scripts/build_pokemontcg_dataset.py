"""
Incremental Pokémon TCG dataset builder.

- Uses /v2/cards with pagination.
- Caches each page's JSON under data/pokemontcg_raw/pages/page_<N>.json
- On rerun, reuses cached pages and only fetches missing pages.
- Skips pages that error (404/429/5xx/timeout) and moves on.
- Slower request timing to be gentle on the API.
- Rebuilds images.csv from all cached pages each run.
"""

import os
import csv
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ.get("POKEMONTCG_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing POKEMONTCG_API_KEY env var. Set it via:\n"
        '    setx POKEMONTCG_API_KEY "your_key_here"'
    )

BASE_URL = "https://api.pokemontcg.io/v2"
HEADERS = {"X-Api-Key": API_KEY}

MIN_RELEASE_DATE = datetime(2015, 1, 1)  # last ~10 years

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "pokemontcg_raw"
IMAGES_ROOT = DATA_ROOT / "images"
PAGES_ROOT = DATA_ROOT / "pages"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
PAGES_ROOT.mkdir(parents=True, exist_ok=True)

PAGE_SIZE = 250
MAX_PAGES = 80              # up to ~20k cards if API behaves
REQUEST_DELAY = 1.0         # slower = nicer to API
IMG_DELAY = 0.1             # between image downloads

# ============================================================
# HELPERS
# ============================================================

def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


def try_fetch_page(page: int):
    """
    Try to fetch a page of cards from the API.
    Returns list of card dicts, or None on error.
    """
    url = f"{BASE_URL}/cards"
    params = {"page": page, "pageSize": PAGE_SIZE}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=25)
    except Exception as e:
        print(f"[WARN] Page {page} request error: {e}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] Page {page} returned {resp.status_code}; skipping.")
        return None

    try:
        body = resp.json()
        return body.get("data", [])
    except Exception as e:
        print(f"[WARN] Page {page} JSON decode error: {e}")
        return None


def load_cached_page(page: int):
    """Load a cached page JSON if it exists, else None."""
    path = PAGES_ROOT / f"page_{page}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read cached page {page}: {e}")
        return None


def save_cached_page(page: int, cards):
    """Save JSON for a page."""
    path = PAGES_ROOT / f"page_{page}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cards, f)
    except Exception as e:
        print(f"[WARN] Failed to save page {page} cache: {e}")


def parse_release_date(card):
    set_info = card.get("set") or {}
    rd = set_info.get("releaseDate")
    if not rd:
        return None
    try:
        return datetime.strptime(rd, "%Y/%m/%d")
    except:
        return None


def download_image(url: str, dest: Path) -> bool:
    """Download image if not already present."""
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, stream=True, timeout=25)
        if resp.status_code != 200:
            print(f"[WARN] Failed image {url}: {resp.status_code}")
            return False

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True

    except Exception as e:
        print(f"[WARN] Image error {url}: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------
    # Build card pool from cached + new pages
    # --------------------------
    all_cards = {}
    print(">>> Loading cached pages + fetching missing pages")

    for page in range(1, MAX_PAGES + 1):
        cached = load_cached_page(page)
        if cached is not None:
            if cached:
                print(f"[INFO] Page {page}: loaded {len(cached)} cards from cache")
            else:
                print(f"[INFO] Page {page}: cached as empty")
            for c in cached:
                cid = c.get("id")
                if cid:
                    all_cards[cid] = c
            continue

        # No cache -> try API
        batch = try_fetch_page(page)
        if batch is None:
            # API failed; leave this page uncached so future runs can retry
            continue

        # Save batch (even if empty)
        save_cached_page(page, batch)
        print(f"[INFO] Page {page}: fetched {len(batch)} cards from API")

        for c in batch:
            cid = c.get("id")
            if cid:
                all_cards[cid] = c

        time.sleep(REQUEST_DELAY)

    print(f">>> Total unique cards gathered (raw): {len(all_cards)}")

    # --------------------------
    # Filter by release date
    # --------------------------
    filtered = []
    for card in all_cards.values():
        rd = parse_release_date(card)
        if rd and rd >= MIN_RELEASE_DATE:
            filtered.append(card)

    print(f">>> Cards released >= {MIN_RELEASE_DATE.date()}: {len(filtered)}")

    # --------------------------
    # Download images + build rows
    # --------------------------
    rows = []

    for card in tqdm(filtered, desc="Downloading images"):
        images = card.get("images") or {}
        img_url = images.get("large") or images.get("small")
        if not img_url:
            continue

        set_info = card.get("set") or {}

        row = {
            "card_id": card.get("id", ""),
            "set_id": set_info.get("id", ""),
            "set_name": set_info.get("name", ""),
            "release_date": set_info.get("releaseDate", ""),
            "number": card.get("number", ""),
            "name": card.get("name", ""),
            "rarity": card.get("rarity", ""),
            "supertype": card.get("supertype", ""),
            "subtypes": ", ".join(card.get("subtypes", []) or []),
            "artist": card.get("artist", ""),
            "image_url": img_url,
        }

        filename = slugify(f"{row['number']}_{row['name']}") + ".jpg"
        dest = IMAGES_ROOT / row["set_id"] / filename
        row["image_path"] = str(dest.as_posix())

        if download_image(img_url, dest):
            rows.append(row)

        time.sleep(IMG_DELAY)

    # --------------------------
    # Write CSV (always overwrite, but built from full card pool)
    # --------------------------
    csv_path = DATA_ROOT / "images.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("\n>>> DONE")
        print(f"Images saved to: {IMAGES_ROOT}")
        print(f"CSV saved to:    {csv_path}")
        print(f"Total usable images: {len(rows)}")
    else:
        print("\n[WARN] No rows were generated; CSV not written.")


if __name__ == "__main__":
    main()