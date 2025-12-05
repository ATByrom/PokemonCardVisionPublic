import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

BASE_URL = "https://www.pricecharting.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DATA_ROOT = Path("data") / "web_raw"

# Curated mix of popular / recent sets (roughly last ~10 years)
SET_URLS = {
    # XY / Evolutions-era (2016)
    "evolutions": "https://www.pricecharting.com/console/pokemon-evolutions",
    # Sun & Moon base (2017)
    "sm-base": "https://www.pricecharting.com/console/pokemon-sun-%26-moon",
    # Sword & Shield-era
    "vivid": "https://www.pricecharting.com/console/pokemon-vivid-voltage",
    "evs": "https://www.pricecharting.com/console/pokemon-evolving-skies",
    "brs": "https://www.pricecharting.com/console/pokemon-brilliant-stars",
    "lost-origin": "https://www.pricecharting.com/console/pokemon-lost-origin",
    "silver-tempest": "https://www.pricecharting.com/console/pokemon-silver-tempest",
    "crz": "https://www.pricecharting.com/console/pokemon-crown-zenith",
    # Scarlet & Violet-era
    "sv-base": "https://www.pricecharting.com/console/pokemon-scarlet-%26-violet",
    "paldea-evolved": "https://www.pricecharting.com/console/pokemon-paldea-evolved",
    "obsidian-flames": "https://www.pricecharting.com/console/pokemon-obsidian-flames",
    "paradox-rift": "https://www.pricecharting.com/console/pokemon-paradox-rift",
    "paldean-fates": "https://www.pricecharting.com/console/pokemon-paldean-fates",
    # you can add/remove here if needed
}

REQUEST_DELAY_SET = 1.0   # delay between set pages
REQUEST_DELAY_CARD = 0.5  # delay between card image downloads

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def safe_get(url: str, *, timeout: int = 20, max_retries: int = 3):
    """
    GET with simple retry + backoff for 429.
    """
    for attempt in range(max_retries):
        try:
            resp = SESSION.get(url, timeout=timeout)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"[WARN] 429 Too Many Requests for {url} — sleeping {wait}s, retry {attempt+1}/{max_retries}")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"[WARN] GET {url} failed on attempt {attempt+1}/{max_retries}: {e}")
            # small backoff on other errors too
            time.sleep(3 * (attempt + 1))
    print(f"[ERROR] Giving up on {url}")
    return None


# ---------------------------------------------------------------------
# Scraping PriceCharting
# ---------------------------------------------------------------------

def parse_pricecharting_set(set_code: str, url: str):
    """
    Parse a PriceCharting 'console/...' page and return a list of cards:
      { 'set_code', 'set_name', 'name', 'number', 'detail_url' }
    """
    print(f"\n>>> Parsing set: {set_code}")
    print(f"    URL: {url}")
    resp = safe_get(url)
    if resp is None:
        print("  Found 0 cards (request failed)")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try to get a nice human-readable set name from <h1>
    h1 = soup.find("h1")
    set_name = h1.get_text(strip=True) if h1 else set_code

    cards = []
    seen = set()

    # Card links look like /game/pokemon-<set>/<card-name-xxx>
    for a in soup.select('a[href^="/game/"]'):
        href = a.get("href", "")
        if "pokemon-" not in href:
            continue
        if "pokemon-cards" in href:
            continue

        label = a.get_text(strip=True)
        if "#" not in label:
            continue  # sealed products etc.

        try:
            name_part, num_part = label.rsplit("#", 1)
        except ValueError:
            continue

        name = name_part.strip()
        number = num_part.strip()

        key = (name, number)
        if key in seen:
            continue
        seen.add(key)

        detail_url = urljoin(BASE_URL, href)

        cards.append(
            {
                "set_code": set_code,
                "set_name": set_name,
                "name": name,
                "number": number,
                "detail_url": detail_url,
            }
        )

    print(f"  Found {len(cards)} cards")
    return cards


def get_card_image_url(card_detail_url: str):
    """
    For a single card page, grab the main card image URL.

    Look for an <img> whose src contains 'game-images' or 'image/'.
    """
    resp = safe_get(card_detail_url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    img = soup.select_one('img[src*="game-images"]')
    if img is None:
        img = soup.select_one('img[src*="image"]')

    if not img:
        print(f"  [WARN] No image tag found on {card_detail_url}")
        return None

    src = img.get("src")
    if not src:
        print(f"  [WARN] Image tag without src on {card_detail_url}")
        return None

    return urljoin(BASE_URL, src)


def download_image(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = safe_get(url)
    if resp is None:
        return False

    try:
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  [WARN] Failed to save {out_path}: {e}")
        return False


# ---------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    all_cards = []

    # 1) Collect card metadata from each curated set
    print("Sets that will be scraped:")
    for code, url in SET_URLS.items():
        print(f"  - {code}: {url}")

    for set_code, url in SET_URLS.items():
        cards = parse_pricecharting_set(set_code, url)
        all_cards.extend(cards)
        time.sleep(REQUEST_DELAY_SET)

    print(f"\nTotal unique card entries parsed: {len(all_cards)}\n")

    rows_for_csv = []

    # 2) For each card, visit its detail page and download the card image
    for card in tqdm(all_cards, desc="Cards"):
        set_code = card["set_code"]
        set_name = card["set_name"]
        name = card["name"]
        number = card["number"]
        detail_url = card["detail_url"]

        # Folder: data/web_raw/<set_code>/###_<name-slug>/
        card_dir = DATA_ROOT / set_code / f"{number.zfill(3)}_{slugify(name)}"
        img_path = card_dir / "pricecharting.jpg"

        if img_path.exists():
            rows_for_csv.append(
                {
                    "set_code": set_code,
                    "set_name": set_name,
                    "name": name,
                    "number": number,
                    "detail_url": detail_url,
                    "image_path": str(img_path),
                }
            )
            continue

        img_url = get_card_image_url(detail_url)
        if not img_url:
            continue

        ok = download_image(img_url, img_path)
        if not ok:
            continue

        rows_for_csv.append(
            {
                "set_code": set_code,
                "set_name": set_name,
                "name": name,
                "number": number,
                "detail_url": detail_url,
                "image_path": str(img_path),
            }
        )

        time.sleep(REQUEST_DELAY_CARD)

    # 3) Save metadata CSV
    csv_path = DATA_ROOT / "cards.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["set_code", "set_name", "name", "number", "detail_url", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"\nSaved metadata to {csv_path}")
    print(f"Total cards with images: {len(rows_for_csv)}")


if __name__ == "__main__":
    main()