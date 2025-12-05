import re
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from PIL import Image
from pillow_heif import register_heif_opener
import pytesseract

# -------------------------------------------------------
# TESSERACT CONFIG (adjust path if needed)
# -------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CONFIG = r"--psm 6 --oem 3"

# Enable HEIC/HEIF support
register_heif_opener()

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384

# Thresholds
UNKNOWN_THRESHOLD = 0.90      # classifier conf below this => unknown
PROTO_MIN_SIM = 0.70          # NEW: proto >= 0.70 can also accept as known
REF_SIM_THRESHOLD = 0.55      # nearest ref card sim must be >= this

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = BASE_DIR / "data" / "experiments" / "pokemon_real_vit_best.pt"
REFERENCE_EMB_PATH = BASE_DIR / "data" / "pokemon_tcg_hf" / "reference_embeddings.pt"
PROTOTYPE_EMB_PATH = BASE_DIR / "data" / "experiments" / "pokemon_class_prototypes.pt"
TEST_DIR = BASE_DIR / "data" / "pokemon_photos" / "test_manual"


# -------------------------------------------------------
# TRANSFORMS
# -------------------------------------------------------

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


# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------

def load_classifier_model():
    """
    Rebuild ViT-B/16 SWAG classifier and load your fine-tuned head.
    """
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    class_names = ckpt["class_names"]
    print("[INFO] Loaded checkpoint with classes:", class_names)

    swag_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=swag_weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, len(class_names))

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def build_feature_model():
    """
    ViT-B/16 SWAG encoder that outputs embeddings instead of logits.
    """
    swag_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = models.vit_b_16(weights=swag_weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model


# -------------------------------------------------------
# REFERENCE INDEX (10k cards)
# -------------------------------------------------------

def load_reference_index():
    if not REFERENCE_EMB_PATH.exists():
        print(f"[WARN] Reference embeddings not found at {REFERENCE_EMB_PATH}")
        return None, None

    data = torch.load(REFERENCE_EMB_PATH, map_location="cpu")
    ref_feats = data["features"]
    ref_meta = data["meta"]

    if not isinstance(ref_feats, torch.Tensor):
        ref_feats = torch.tensor(ref_feats, dtype=torch.float32)

    n_feats = ref_feats.shape[0]
    n_meta = len(ref_meta)
    if n_feats != n_meta:
        print(f"[WARN] features/meta length mismatch ({n_feats} vs {n_meta}); "
              f"truncating to {min(n_feats, n_meta)}")
        n = min(n_feats, n_meta)
        ref_feats = ref_feats[:n]
        ref_meta = ref_meta[:n]

    print(f"[INFO] Loaded reference index: {len(ref_meta)} cards, dim={ref_feats.shape[1]}")
    return ref_feats, ref_meta


def nearest_reference_card(query_emb, ref_feats, ref_meta, top_k: int = 8):
    """
    query_emb: 1D tensor [D] on DEVICE
    ref_feats: [N, D] tensor on CPU (we'll move to DEVICE)

    Returns:
        best_meta, best_sim, best_idx, topk_candidates

        where topk_candidates is a list of (meta, sim, idx) tuples
        for the top_k nearest reference cards.
    """
    if ref_feats is None or ref_meta is None:
        return None, None, None, []

    # normalize query
    q = query_emb / query_emb.norm()

    # move features to DEVICE and normalize
    feats = ref_feats.to(DEVICE)
    feats = feats / feats.norm(dim=1, keepdim=True)

    sims = torch.matmul(feats, q)  # [N] cosine-like scores
    best_val, best_idx = sims.max(dim=0)

    # top-K candidates for re-ranking
    k = min(top_k, sims.shape[0])
    top_vals, top_idx = torch.topk(sims, k=k)

    topk_candidates = []
    for v, i in zip(top_vals, top_idx):
        i_int = int(i)
        topk_candidates.append((ref_meta[i_int], float(v), i_int))

    return ref_meta[int(best_idx)], float(best_val), int(best_idx), topk_candidates
# -------------------------------------------------------
# PROTOTYPES (PER-CLASS)
# -------------------------------------------------------

def load_prototypes():
    if not PROTOTYPE_EMB_PATH.exists():
        print(f"[WARN] Prototype file not found at {PROTOTYPE_EMB_PATH}")
        return None

    data = torch.load(PROTOTYPE_EMB_PATH, map_location="cpu")
    protos_raw = data.get("prototypes", {})

    prototypes = {}
    for label, vec in protos_raw.items():
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)
        # ensure 1D
        vec = vec.flatten()
        # normalize
        vec = vec / vec.norm()
        prototypes[label] = vec.to(DEVICE)

    print(f"[INFO] Loaded {len(prototypes)} class prototypes")
    return prototypes


def prototype_similarity(query_emb, prototypes, label):
    if prototypes is None:
        return None
    if label not in prototypes:
        return None

    q = query_emb / query_emb.norm()
    proto = prototypes[label]
    return float(torch.dot(q, proto))


# -------------------------------------------------------
# OCR FOR UNKNOWN CARDS
# -------------------------------------------------------

def extract_name_and_number(pil_img):
    """
    Simple OCR to guess a card name and number like '064/203'.
    """
    try:
        text = pytesseract.image_to_string(pil_img, config=TESSERACT_CONFIG)
        text = text.replace("\n", " ")

        # card number pattern: e.g. "064/203"
        m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", text)
        num = m.group(0) if m else None

        # heuristic: first Title-case token as name
        name = None
        for tok in text.split():
            if tok.istitle() and len(tok) > 3:
                name = tok
                break

        # filter out tiny garbage words
        if name and len(name) < 4:
            name = None

        return name, num
    except Exception:
        return None, None


# -------------------------------------------------------
# MAIN PREDICTION LOOP
# -------------------------------------------------------

def main():
    if not TEST_DIR.exists():
        raise SystemExit(f"Test folder not found: {TEST_DIR.resolve()}")

    exts = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
    image_paths = sorted(
        [p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts]
    )
    if not image_paths:
        raise SystemExit(f"No images found in {TEST_DIR.resolve()}")

    transform = build_val_transform()
    clf_model, class_names = load_classifier_model()
    feat_model = build_feature_model()

    ref_feats, ref_meta = load_reference_index()
    prototypes = load_prototypes()

    # Build a set of "known" Pokémon-like names from the 10k index
    pokemon_name_set = set()
    if ref_meta is not None:
        for m in ref_meta:
            nm = m.get("name", "")
            if not nm:
                continue
            first = nm.split()[0]
            if first:
                pokemon_name_set.add(first)

    print(f"[INFO] Found {len(image_paths)} test images in {TEST_DIR}:\n")

    for p in image_paths:
        img = Image.open(p).convert("RGB")

        # ----- Prepare tensor once -----
        img_t = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Classifier forward
            logits = clf_model(img_t)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            conf = float(conf)
            label = class_names[int(idx)]

            # Feature forward
            emb = feat_model(img_t)[0]  # [D]

        # ----- Prototype similarity (used as backup) -----
        proto_sim = prototype_similarity(emb, prototypes, label)

        # Decision: known vs unknown
        # NEW: accept as known if classifier is confident OR
        #      prototype similarity says "this really looks like that class".
        is_known = (conf >= UNKNOWN_THRESHOLD) or (
            proto_sim is not None and proto_sim >= PROTO_MIN_SIM
        )

        if is_known:
            # Training-set cards: just trust the classifier + prototypes
            print(
                f"{p.name:25s} -> {label:22s} "
                f"(conf {conf:.3f}, proto_sim={proto_sim if proto_sim is not None else 'n/a'})"
            )
            continue

        # ----- UNKNOWN PATH: retrieval from 10k + OCR-aware re-ranking -----
        best_meta, best_sim, _, candidates = nearest_reference_card(
            emb, ref_feats, ref_meta, top_k=8
        )

        # OCR for name/number (used only to disambiguate, not to force a match)
        ocr_name, ocr_num = extract_name_and_number(img)
        ocr_name_clean = ocr_name.strip().lower() if ocr_name else None
        ocr_num_clean = ocr_num.strip().lower() if ocr_num else None

        chosen_meta = best_meta
        chosen_sim = best_sim

        # If OCR gave us something useful, try to pick the *best* candidate
        # among the top-K neighbors rather than blindly trusting the global best.
        if candidates and (ocr_name_clean or ocr_num_clean):
            filtered = candidates

            # 1) If we got a name from OCR, restrict to that exact name (case-insensitive)
            if ocr_name_clean:
                name_filtered = [
                    (m, s, i)
                    for (m, s, i) in filtered
                    if m.get("name", "").strip().lower() == ocr_name_clean
                ]
                if name_filtered:
                    filtered = name_filtered

            # 2) If we also got a number (e.g. '98/159', '112', 'XY112'),
            #    prefer candidates whose number string contains that pattern.
            if ocr_num_clean and filtered:
                num_pref = [
                    (m, s, i)
                    for (m, s, i) in filtered
                    if ocr_num_clean in str(m.get("number", "")).lower()
                ]
                if num_pref:
                    filtered = num_pref

            # 3) Among whatever is left, take the one with highest similarity
            if filtered:
                chosen_meta, chosen_sim, _ = max(filtered, key=lambda x: x[1])

        # Decide whether we trust this retrieval at all
        if chosen_meta is not None and chosen_sim is not None and chosen_sim >= REF_SIM_THRESHOLD:
            ref_name = chosen_meta.get("name", "")
            ref_number = chosen_meta.get("number", "")
            ref_set = chosen_meta.get("set_name", "") or chosen_meta.get("set", "")
            extra = f" ({ref_set})" if ref_set else ""

            print(
                f"{p.name:25s} -> UNKNOWN (conf {conf:.3f})  "
                f"{ref_name} {ref_number}{extra} is not currently in the data set "
                f"(best match sim={chosen_sim:.3f})"
            )
        else:
            # Soft failure: nothing in the 10k index was a good enough match,
            # or OCR was too messy to help.
            msg = (
                "Card not recognized as part of the data set, "
                "take a picture as close as you can with all corners and edges in the frame."
            )
            if ocr_name or ocr_num:
                # You *can* optionally include the OCR guess for debugging, but it's noisy.
                print(f"{p.name:25s} -> UNKNOWN (conf {conf:.3f})  {msg}")
            else:
                print(f"{p.name:25s} -> UNKNOWN (conf {conf:.3f})  {msg}")

if __name__ == "__main__":
    main()