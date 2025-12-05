"""
augment_with_ebay_images.py

Reads data/web_raw/cards.csv (from build_pokemon_web_dataset.py),
revisits each PriceCharting card page, grabs a few eBay listing links,
then downloads the main image from each eBay listing.

Images are saved next to the existing PriceCharting scan, e.g.:

  data/web_raw/<set_code>/###_<card-name>/
      pricecharting.jpg
      ebay_00.jpg
      ebay_01.jpg
      ...

Also writes data/web_raw/images.csv with one row per image
(both PriceCharting + eBay) for training.

DEPENDENCIES:
    pip install requests beautifulsoup4 tqdm
"""

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

DATA_ROOT = Path("data") / "web_raw"
CARDS_CSV = DATA_ROOT / "cards.csv"
IMAGES_CSV = DATA_ROOT / "images.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Max number of ebay listings to pull images from per card
MAX_EBAY_PER_CARD = 3

# Delays between requests so we’re not jerks
REQUEST_DELAY_PC_CARD = 0.4
REQUEST_DELAY_EBAY = 0.5

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_get(url: str, *, timeout: int = 20):
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"[WARN] GET {url} failed: {e}")
        return None


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def extract_ebay_links_from_pc_card_page(html: str, max_links: int):
    """
    Given the HTML of a PriceCharting card page, grab up to max_links
    eBay URLs from the 'Sold Listings' table.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    seen = set()

    # Any <a> whose href contains "www.ebay.com"
    for a in soup.select('a[href*="www.ebay.com"]'):
        href = a.get("href")
        if not href:
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
        if len(links) >= max_links:
            break

    return links


def get_ebay_main_image_url(ebay_url: str):
    """
    Fetch an eBay listing and try to grab the main item image.

    Heuristic: look for:
      - <img id="icImg"> (common on eBay item pages)
      - or any <img> whose src contains 'i.ebayimg.com'
    """
    resp = safe_get(ebay_url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1) Try the standard main image
    img = soup.select_one("img#icImg")
    if img and img.get("src"):
        return img["src"]

    # 2) Fallback: any ebay image host
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "i.ebayimg.com" in src:
            return src

    print(f"  [WARN] Could not find main image on eBay page: {ebay_url}")
    return None


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
# Main
# ---------------------------------------------------------------------

def main():
    if not CARDS_CSV.exists():
        print(f"[ERROR] {CARDS_CSV} not found. Run build_pokemon_web_dataset.py first.")
        return

    # Read card-level metadata
    cards = []
    with open(CARDS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards.append(row)

    print(f"Loaded {len(cards)} cards from {CARDS_CSV}")

    # Prepare list of image rows (one per image)
    image_rows = []

    # 1) Add existing PriceCharting images as rows
    for card in cards:
        pc_img_path = card["image_path"]
        image_rows.append(
            {
                "set_code": card.get("set_code", ""),
                "year": card.get("year", ""),
                "set_name": card.get("set_name", ""),
                "name": card["name"],
                "number": card["number"],
                "source": "pricecharting",
                "image_path": pc_img_path,
                "detail_url": card["detail_url"],
                "ebay_url": "",
            }
        )

    # 2) For each card, try to add some eBay images
    for card in tqdm(cards, desc="Cards (ebay images)"):
        detail_url = card["detail_url"]
        set_code = card.get("set_code", "")
        set_name = card.get("set_name", "")
        year = card.get("year", "")
        name = card["name"]
        number = card["number"]

        # Card directory is inferred from the PriceCharting image path
        pc_img_path = Path(card["image_path"])
        card_dir = pc_img_path.parent

        # Fetch the PriceCharting card page
        resp = safe_get(detail_url)
        if resp is None:
            continue

        ebay_links = extract_ebay_links_from_pc_card_page(
            resp.text, max_links=MAX_EBAY_PER_CARD
        )

        if not ebay_links:
            time.sleep(REQUEST_DELAY_PC_CARD)
            continue

        for idx, ebay_url in enumerate(ebay_links):
            img_url = get_ebay_main_image_url(ebay_url)
            if not img_url:
                time.sleep(REQUEST_DELAY_EBAY)
                continue

            out_path = card_dir / f"ebay_{idx:02d}.jpg"
            ok = download_image(img_url, out_path)
            if not ok:
                time.sleep(REQUEST_DELAY_EBAY)
                continue

            image_rows.append(
                {
                    "set_code": set_code,
                    "year": year,
                    "set_name": set_name,
                    "name": name,
                    "number": number,
                    "source": "ebay",
                    "image_path": str(out_path),
                    "detail_url": detail_url,
                    "ebay_url": ebay_url,
                }
            )

            time.sleep(REQUEST_DELAY_EBAY)

        time.sleep(REQUEST_DELAY_PC_CARD)

    # 3) Write out images.csv
    with open(IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "set_code",
                "year",
                "set_name",
                "name",
                "number",
                "source",
                "image_path",
                "detail_url",
                "ebay_url",
            ],
        )
        writer.writeheader()
        writer.writerows(image_rows)

    print(f"\nWrote {len(image_rows)} image rows to {IMAGES_CSV}")


if __name__ == "__main__":
    main()