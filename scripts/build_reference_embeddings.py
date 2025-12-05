import csv
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from pillow_heif import register_heif_opener
from torchvision import models, transforms

# Enable HEIC just in case
register_heif_opener()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 384

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
CARDS_CSV = BASE_DIR / "data" / "pokemon_tcg_hf" / "cards.csv"
IMAGES_ROOT = BASE_DIR  # local_path in csv is already "data/..."
OUT_PATH = BASE_DIR / "data" / "pokemon_tcg_hf" / "reference_embeddings.pt"


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_feature_model():
    # Same weights as everywhere else (SWAG E2E)
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Identity()  # output embeddings, not logits
    model.to(DEVICE)
    model.eval()
    return model


def main():
    if not CARDS_CSV.exists():
        raise SystemExit(f"cards.csv not found: {CARDS_CSV}")

    transform = build_transform()
    model = build_feature_model()

    features = []
    meta = []

    total_rows = 0
    processed = 0
    skipped_missing = 0
    skipped_errors = 0

    print(f"[INFO] Reading card metadata from: {CARDS_CSV}")

    with CARDS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_rows = len(rows)
    print(f"[INFO] Found {total_rows} rows in cards.csv")

    for i, row in enumerate(rows, start=1):
        local_path = row.get("local_path", "").strip()
        if not local_path:
            skipped_missing += 1
            continue

        img_path = IMAGES_ROOT / local_path
        if not img_path.exists():
            skipped_missing += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model(img_t)  # [1, D]

            features.append(emb.cpu()[0])  # [D]

            meta.append({
                "id": row.get("id", ""),
                "name": row.get("name", ""),
                "number": row.get("number", ""),
                "set_id": row.get("set_id", ""),
                "set_name": row.get("set_name", ""),
                "set_release_date": row.get("set_release_date", ""),
                "local_path": local_path,
            })

            processed += 1

        except Exception as e:
            skipped_errors += 1
            print(f"[WARN] Failed on {img_path}: {e}")

        if i % 500 == 0:
            print(f"[INFO] Processed {i}/{total_rows} rows ...")

    if not features:
        raise SystemExit("[ERROR] No embeddings were created; aborting.")

    feats_tensor = torch.stack(features, dim=0)  # [N, D]
    out = {
        "features": feats_tensor,
        "meta": meta,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_PATH)

    print(f"[INFO] Saved embeddings to: {OUT_PATH}")
    print(f"[INFO] Total rows:      {total_rows}")
    print(f"[INFO] With image:      {processed}")
    print(f"[INFO] Missing image:   {skipped_missing}")
    print(f"[INFO] Errors:          {skipped_errors}")
    print(f"[INFO] Final shapes:    features={feats_tensor.shape}, meta={len(meta)}")


if __name__ == "__main__":
    main()