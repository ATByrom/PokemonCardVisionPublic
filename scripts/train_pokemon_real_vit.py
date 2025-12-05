import os
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()
import pandas as pd
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 384  # must match the ViT weights' expected image size
BATCH_SIZE = 16
EPOCHS = 7
LR = 3e-5
WEIGHT_DECAY = 0.01

LABELS_CSV = Path("data/pokemon_photos/labels.csv")
RAW_DIR = Path("data/pokemon_photos/raw")
METRICS_CSV = Path("data/experiments/pokemon_real_vit_metrics.csv")
CHECKPOINT_PATH = Path("data/experiments/pokemon_real_vit_best.pt")


class PokemonPhotoDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label


def make_datasets():
    df = pd.read_csv(LABELS_CSV)
    # drop any "skip" rows just in case
    df = df[df["label"] != "skip"].reset_index(drop=True)

    df["filepath"] = df["filename"].apply(lambda fn: RAW_DIR / fn)
    before = len(df)
    df = df[df["filepath"].apply(lambda p: p.exists())].reset_index(drop=True)
    df = df.drop(columns=["filepath"])
    print(f"[INFO] Filtered out {before - len(df)} rows with missing image files")

    # stratified split by label
    rng = np.random.default_rng(42)
    train_indices = []
    val_indices = []

    for label, group in df.groupby("label"):
        idxs = group.index.to_numpy()
        rng.shuffle(idxs)
        cut = int(len(idxs) * 0.8)
        train_indices.extend(idxs[:cut])
        val_indices.extend(idxs[cut:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)

    print(f"[INFO] Total labelled: {len(df)}")
    print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"[INFO] Classes: {sorted(df['label'].unique())}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(
            IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = PokemonPhotoDataset(train_df, RAW_DIR, transform=train_transform)
    val_ds = PokemonPhotoDataset(val_df, RAW_DIR, transform=val_transform)

    return train_ds, val_ds, sorted(df["label"].unique())


def build_model(num_classes: int):
    # Use same weights as the other script (adjust if you changed it)
    weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=weights)

    # replace classifier head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    print(f"[INFO] Using device: {DEVICE}")
    train_ds, val_ds, class_names = make_datasets()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Train loss: {train_loss:.4f}  |  acc: {train_acc:.2f}%")
        print(f"Val   loss: {val_loss:.4f}  |  acc: {val_acc:.2f}%")

        with METRICS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # simple checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, CHECKPOINT_PATH)
            print(f"[INFO] New best model saved with val_acc={best_val_acc:.2f}%")

    print(f"\n[INFO] Training complete. Best val_acc = {best_val_acc:.2f}%")
    print(f"[INFO] Metrics saved to: {METRICS_CSV}")
    print(f"[INFO] Best checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()