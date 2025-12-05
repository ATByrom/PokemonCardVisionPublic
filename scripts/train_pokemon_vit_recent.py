import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms, models


# -----------------------------
# Config
# -----------------------------
SEED = 42
VAL_FRACTION = 0.1          # 10% of data for validation
BATCH_SIZE = 64
NUM_EPOCHS = 8              # adjust if you want more/less overnight
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4             # you can lower this on Windows if needed
IMAGE_SIZE = 224


# -----------------------------
# Utility: seeding
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class PokemonCardDataset(Dataset):
    def __init__(self, rows: List[Dict], id_to_idx: Dict[str, int], root: Path, transform=None):
        self.rows = rows
        self.id_to_idx = id_to_idx
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        # local_path is assumed to be relative to repo root
        local_path = row.get("local_path", "")
        img_path = self.root / local_path

        # Load image
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        card_id = row["id"]
        label = self.id_to_idx[card_id]
        return img, label


# -----------------------------
# Load metadata from cards.csv
# -----------------------------
def load_card_metadata(root: Path) -> Tuple[List[Dict], Dict[str, int], Dict[int, str], Dict[str, Dict]]:
    cards_csv = root / "data" / "pokemon_tcg_hf" / "cards.csv"
    if not cards_csv.exists():
        raise FileNotFoundError(f"cards.csv not found at {cards_csv}")

    rows = []
    with cards_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Basic sanity checks: need id + local_path not empty and file must exist
            cid = r.get("id", "").strip()
            local_path = r.get("local_path", "").strip()
            if not cid or not local_path:
                continue
            img_path = root / local_path
            if not img_path.exists():
                # skip rows that don't actually have a downloaded image
                continue
            rows.append(r)

    if not rows:
        raise RuntimeError("No valid rows with images found in cards.csv")

    # Unique card IDs
    unique_ids = sorted({r["id"] for r in rows})
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
    idx_to_id = {i: cid for cid, i in id_to_idx.items()}

    # One metadata dict per card_id (first occurrence wins)
    meta_by_id: Dict[str, Dict] = {}
    for r in rows:
        cid = r["id"]
        if cid not in meta_by_id:
            meta_by_id[cid] = {
                "id": cid,
                "name": r.get("name", ""),
                "set_id": r.get("set_id", ""),
                "set_name": r.get("set_name", ""),
                "set_release_date": r.get("set_release_date", ""),
                "number": r.get("number", ""),
                "local_path": r.get("local_path", ""),
            }

    print(f"[INFO] Loaded {len(rows)} image rows for training/validation")
    print(f"[INFO] Unique cards (classes): {len(unique_ids)}")

    return rows, id_to_idx, idx_to_id, meta_by_id


# -----------------------------
# Transforms
# -----------------------------
def build_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_tf, val_tf


# -----------------------------
# Build model
# -----------------------------
def build_model(num_classes: int) -> nn.Module:
    VitWeights = models.ViT_B_16_Weights
    # Use standard 224x224 ImageNet weights
    weights = VitWeights.IMAGENET1K_V1

    model = models.vit_b_16(weights=weights)

    # Sanity check (optional): print expected image size
    print(f"[INFO] ViT expected image size: {model.image_size}")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 20 == 0:
            avg_loss = running_loss / total if total > 0 else 0.0
            acc = 100.0 * correct / total if total > 0 else 0.0
            print(f"  [Train] Step {step+1}/{len(loader)}  Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(SEED)

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load metadata
    rows, id_to_idx, idx_to_id, meta_by_id = load_card_metadata(root)

    # Shuffle rows before splitting
    random.shuffle(rows)

    num_total = len(rows)
    val_size = max(1, int(num_total * VAL_FRACTION))
    train_size = num_total - val_size

    print(f"[INFO] Total samples: {num_total}  Train: {train_size}  Val: {val_size}")

    # Build transforms & dataset
    train_tf, val_tf = build_transforms(IMAGE_SIZE)

    full_dataset = PokemonCardDataset(rows, id_to_idx=id_to_idx, root=root, transform=None)
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Wrap subsets with transforms by monkey-patching their dataset.transform
    # (random_split keeps references to the same underlying dataset)
    full_dataset.transform = train_tf
    train_subset.dataset.transform = train_tf
    val_subset.dataset.transform = val_tf

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_classes = len(id_to_idx)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    best_val_acc = 0.0
    best_state = None

    print("[INFO] Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"[Epoch {epoch+1}] Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            ckpt_path = checkpoints_dir / "vit_b16_pokemon_best.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "id_to_idx": id_to_idx,
                    "idx_to_id": idx_to_id,
                    "meta_by_id": meta_by_id,
                    "num_classes": num_classes,
                    "image_size": IMAGE_SIZE,
                },
                ckpt_path,
            )
            print(f"[INFO] New best model saved to {ckpt_path} (Val Acc: {best_val_acc:.2f}%)")

    print("\n[INFO] Training complete.")
    print(f"[INFO] Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()