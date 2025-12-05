import csv
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image
from pillow_heif import register_heif_opener

# Enable HEIC/HEIF
register_heif_opener()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_CSV = BASE_DIR / "data" / "pokemon_photos" / "labels.csv"
IMAGES_DIR = BASE_DIR / "data" / "pokemon_photos" / "raw"
PROTOTYPE_EMB_PATH = BASE_DIR / "data" / "experiments" / "pokemon_class_prototypes.pt"

def build_val_transform():
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
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    # identity head → returns embeddings instead of logits
    model.heads.head = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model

def main():
    if not LABELS_CSV.exists():
        raise SystemExit(f"labels.csv not found at {LABELS_CSV.resolve()}")

    if not IMAGES_DIR.exists():
        raise SystemExit(f"Image folder not found: {IMAGES_DIR.resolve()}")

    transform = build_val_transform()
    model = build_feature_model()

    sums = {}
    counts = {}

    with LABELS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename") or row.get("image") or row.get("file")
            label = row.get("label")
            if not fname or not label:
                continue

            img_path = IMAGES_DIR / fname
            if not img_path.exists():
                print(f"[WARN] Missing image: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model(img_t)[0]  # [D]

            if label not in sums:
                sums[label] = emb.clone()
                counts[label] = 1
            else:
                sums[label] += emb
                counts[label] += 1

    prototypes = {}
    for label, vec_sum in sums.items():
        n = counts[label]
        proto = vec_sum / n
        proto = proto / proto.norm()
        prototypes[label] = proto.cpu()
        print(f"[INFO] Prototype for {label}: {n} images")

    PROTOTYPE_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": prototypes}, PROTOTYPE_EMB_PATH)
    print(f"[INFO] Saved {len(prototypes)} prototypes to {PROTOTYPE_EMB_PATH.resolve()}")

if __name__ == "__main__":
    main()