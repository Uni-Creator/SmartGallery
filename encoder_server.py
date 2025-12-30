import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import clip
import faiss
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAISS_INDEX_PATH = Path("embeddings/image_faiss_index.idx")
EMBEDDINGS_PATH = Path("embeddings/image_embeddings_normalized.npy")
CSV_PATH = Path("data/images_with_captions_and_tags.csv")

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

index = faiss.read_index(str(FAISS_INDEX_PATH))
df = pd.read_csv(CSV_PATH)
image_paths = df["image_path"].tolist()
image_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

def encode_single_image(path: str) -> np.ndarray:
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)

print("ENCODER_READY", flush=True)

while True:
    line = sys.stdin.readline().strip()

    if not line:
        continue

    if line == "EXIT":
        break

    if line == "UPDATE":
        new_vectors = []
        new_paths = []

        while True:
            img = sys.stdin.readline().strip()
            if img == "END_UPDATE":
                break

            if not Path(img).exists():
                print(f"ERROR Image not found: {img}", flush=True)
                new_vectors = []
                break

            vec = encode_single_image(img)
            new_vectors.append(vec)
            new_paths.append(img)

        if not new_vectors:
            continue

        new_vectors = np.vstack(new_vectors)

        index.add(new_vectors)
        image_embeddings = np.vstack([image_embeddings, new_vectors])
        image_paths.extend(new_paths)

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        np.save(EMBEDDINGS_PATH, image_embeddings)

        df = pd.DataFrame({"image_path": image_paths})
        df.to_csv(CSV_PATH, index=False)

        print("ENCODED", flush=True)
