import csv
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from PIL import Image
from pillow_heif import register_heif_opener
import numpy as np
import pandas as pd

# Enable HEIC/HEIF support
register_heif_opener()

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_CSV = BASE_DIR / "data" / "pokemon_photos" / "labels.csv"
RAW_DIR = BASE_DIR / "data" / "pokemon_photos" / "raw"

PROTOTYPE_EMB_PATH = BASE_DIR / "data" / "experiments" / "pokemon_class_prototypes.pt"
PROTOTYPE_IMG_DIR = BASE_DIR / "data" / "experiments" / "prototype_images"

IMG_SIZE = 384  # same as training


# -------------------------------------------------------
# MODEL (FEATURE EXTRACTOR)
# -------------------------------------------------------

def build_feature_model():
    """
    ViT-B/16 SWAG model with the classification head removed.
    Outputs 768-dim embeddings for each image.
    """
    swag_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=swag_weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Identity()  # now outputs embeddings
    model.to(DEVICE)
    model.eval()
    return model


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


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    if not LABELS_CSV.exists():
        raise SystemExit(f"Labels CSV not found: {LABELS_CSV.resolve()}")

    if not RAW_DIR.exists():
        raise SystemExit(f"Raw image folder not found: {RAW_DIR.resolve()}")

    df = pd.read_csv(LABELS_CSV)
    # Drop any 'skip' labels
    df = df[df["label"] != "skip"].reset_index(drop=True)

    print(f"[INFO] Loaded {len(df)} labeled rows from {LABELS_CSV.name}")
    class_names = sorted(df["label"].unique())
    print(f"[INFO] Classes: {class_names}")

    feat_model = build_feature_model()
    transform = build_transform()

    # For embeddings
    embs_by_class = {label: [] for label in class_names}
    # For composite images (just for visualization; same size as IMG_SIZE)
    imgs_by_class = {label: [] for label in class_names}

    total = len(df)
    for idx, row in df.iterrows():
        filename = row["filename"]
        label = row["label"]

        img_path = RAW_DIR / filename
        if not img_path.exists():
            print(f"[WARN] Missing image file: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}")
            continue

        # Embedding
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = feat_model(img_t)  # [1, 768]

        embs_by_class[label].append(emb.cpu())

        # Composite image (resize but no normalization)
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        imgs_by_class[label].append(np.array(img_resized, dtype=np.float32))

        if (idx + 1) % 20 == 0 or idx == total - 1:
            print(f"[INFO] Processed {idx+1}/{total} images")

    # Build prototypes
    PROTOTYPE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    prototypes = {}

    for label in class_names:
        emb_list = embs_by_class[label]
        img_list = imgs_by_class[label]

        if not emb_list:
            print(f"[WARN] No embeddings for class '{label}', skipping")
            continue

        # ----- Embedding prototype -----
        emb_stack = torch.cat(emb_list, dim=0)  # [N, 768]
        proto = emb_stack.mean(dim=0)           # [768]
        proto = proto / proto.norm()            # normalize
        prototypes[label] = proto

        # ----- Composite image (median) -----
        try:
            stack = np.stack(img_list, axis=0)  # [N, H, W, 3]
            median_img = np.median(stack, axis=0).astype(np.uint8)
            pil_proto = Image.fromarray(median_img)
            out_path = PROTOTYPE_IMG_DIR / f"{label}_composite.png"
            pil_proto.save(out_path)
            print(f"[INFO] Saved composite image for '{label}' to {out_path}")
        except Exception as e:
            print(f"[WARN] Failed composite image for '{label}': {e}")

    # Save embedding prototypes
    PROTOTYPE_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes,
            "class_names": class_names,
        },
        PROTOTYPE_EMB_PATH,
    )
    print(f"[INFO] Saved embedding prototypes to {PROTOTYPE_EMB_PATH}")


if __name__ == "__main__":
    main()