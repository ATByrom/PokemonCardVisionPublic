"""
tcgplayer_build_dataset.py

Build a complete Pokémon card training dataset using the official
TCGplayer API (requires your API keys).

Features:
- Authenticate with OAuth2
- Fetch all Pokémon sets
- Fetch all cards in each set
- Download official card images
- Save dataset to: data/tcgplayer_raw/
- Generate images.csv for training

REQUIRES:
    pip install requests tqdm
"""

import os
import csv
import time
import requests
from tqdm import tqdm
from pathlib import Path

# ---------------------------------------------------------------------
# INSERT YOUR API KEYS HERE
# ---------------------------------------------------------------------
PUBLIC_KEY = "YOUR_PUBLIC_KEY"
PRIVATE_KEY = "YOUR_PRIVATE_KEY"

# Alternatively if you have client credentials:
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"

# ---------------------------------------------------------------------

API_BASE = "https://api.tcgplayer.com"
DATA_ROOT = Path("data") / "tcgplayer_raw"
IMAGES_CSV = DATA_ROOT / "images.csv"


def get_access_token():
    """Authenticate with TCGplayer."""
    url = f"{API_BASE}/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    resp = requests.post(url, data=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def get_pokemon_category_id(token):
    """Find the Pokémon category."""
    url = f"{API_BASE}/catalog/categories"
    headers = {"Authorization": f"bearer {token}"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    categories = resp.json()["results"]

    for c in categories:
        if "pokemon" in c["displayName"].lower():
            return c["categoryId"]

    raise ValueError("Pokémon category not found!")


def get_sets(token, category_id):
    """Fetch all sets in Pokémon category."""
    url = f"{API_BASE}/catalog/groups?categoryId={category_id}"
    headers = {"Authorization": f"bearer {token}"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    return resp.json()["results"]


def get_cards_in_set(token, group_id):
    """Fetch all cards for a set."""
    url = f"{API_BASE}/catalog/products?groupId={group_id}&getExtendedFields=true"
    headers = {"Authorization": f"bearer {token}"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["results"]


def download_image(url, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    return False


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print("Authenticating with TCGplayer...")
    token = get_access_token()
    print("✓ Authenticated")

    print("Fetching Pokémon category...")
    category_id = get_pokemon_category_id(token)
    print(f"✓ Pokémon category ID: {category_id}")

    print("Fetching sets...")
    sets = get_sets(token, category_id)
    print(f"✓ Found {len(sets)} sets")

    rows = []

    for s in tqdm(sets, desc="Sets"):
        group_id = s["groupId"]
        set_name = s["name"]

        # Fetch all cards in this set
        try:
            cards = get_cards_in_set(token, group_id)
        except:
            continue

        for card in cards:
            product_id = card["productId"]
            name = card["name"]
            number = card.get("number", "")
            image_url = card.get("imageUrl", None)

            # Skip if no image
            if not image_url:
                continue

            # Local file path
            safe_set = set_name.replace("/", "-")
            safe_name = name.replace("/", "-")

            out_path = DATA_ROOT / safe_set / f"{number}_{safe_name}.jpg"

            ok = download_image(image_url, out_path)
            if not ok:
                continue

            rows.append({
                "set": set_name,
                "name": name,
                "number": number,
                "product_id": product_id,
                "image_path": str(out_path)
            })

            time.sleep(0.1)  # be polite

    # Save master CSV
    with open(IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "name", "number", "product_id", "image_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset complete!")
    print(f"Images saved to {DATA_ROOT}")
    print(f"Metadata CSV: {IMAGES_CSV}")
    print(f"Total images: {len(rows)}")


if __name__ == "__main__":
    main()