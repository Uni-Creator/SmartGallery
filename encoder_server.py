import sys
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import clip
import faiss
import pandas as pd

# Step 1: Paths and device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAISS_INDEX_PATH = Path("embeddings") / "image_faiss_index.idx"
EMBEDDINGS_PATH = Path("embeddings") / "image_embeddings_normalized.npy"
CSV_PATH = Path("data") / "images_with_captions_and_tags.csv"

# Step 2: Load CLIP model once

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# Step 3: Load FAISS index

if not FAISS_INDEX_PATH.exists():
    raise FileNotFoundError("FAISS index not found")

index = faiss.read_index(str(FAISS_INDEX_PATH))

# Step 4: Load metadata + embeddings properly

if not CSV_PATH.exists() or not EMBEDDINGS_PATH.exists():
    raise FileNotFoundError("Metadata or embeddings file missing")

df = pd.read_csv(CSV_PATH)
image_paths = df["image_path"].tolist()

image_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

if len(image_paths) != len(image_embeddings):
    raise ValueError("Mismatch between image paths and embeddings")

# Step 5: Encode image

def encode_image(image_path: str) -> np.ndarray:
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy().astype(np.float32)


print("ENCODER_READY", flush=True)

# Step 6: Live stdin loop

while True:

    image_path = sys.stdin.readline().strip()

    if image_path == "EXIT":
        break

    try:
        new_vector = encode_image(image_path)

        # Add to FAISS
        index.add(new_vector)

        # Update local data
        image_paths.append(image_path)
        image_embeddings = np.vstack([image_embeddings, new_vector])

        # Save updated index & embeddings
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        np.save(EMBEDDINGS_PATH, image_embeddings)

        # Also update CSV
        df = pd.concat([
            df,
            pd.DataFrame({"image_path": [image_path]})
        ], ignore_index=True)

        df.to_csv(CSV_PATH, index=False)

        print("UPDATED", flush=True)
        print("ENCODED", flush=True)

    except Exception as e:
        print("ERROR", str(e), flush=True)
