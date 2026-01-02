import sys
from pathlib import Path
import numpy as np
import torch
import faiss
import pandas as pd
from image_captioning_clip_pipeline import EmbeddingGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAISS_INDEX_PATH = Path("embeddings/image_faiss_index.idx")
EMBEDDINGS_PATH = Path("embeddings/image_embeddings_normalized.npy")
CSV_PATH = Path("data/images_path.csv")

embedder = EmbeddingGenerator(DEVICE)

index = faiss.read_index(str(FAISS_INDEX_PATH))
df = pd.read_csv(CSV_PATH)
image_paths = df["image_path"].tolist()
image_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)


def encode_images(paths: list[str]) -> np.ndarray:
    return embedder.embed_images(paths)

    

print("ENCODER_READY", flush=True)

while True:
    line = sys.stdin.readline().strip()

    if not line:
        continue

    if line == "EXIT":
        break

    if line == "UPDATE":
        new_paths = []

        while True:
            img = sys.stdin.readline().strip()
            if img == "END_UPDATE":
                break

            if not Path(img).exists():
                print(f"ERROR Image not found: {img}", flush=True)
                new_paths = []
                break

            new_paths.append(img)

        if not new_paths:
            continue

        new_embeddings = encode_images(new_paths)

        assert new_embeddings.shape[1] == index.d, "Embedding dimension mismatch"

        index.add(new_embeddings)
        image_embeddings = np.vstack([image_embeddings, new_embeddings])
        image_paths.extend(new_paths)

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        np.save(EMBEDDINGS_PATH, image_embeddings)

        df = pd.DataFrame({"image_path": image_paths})
        df.to_csv(CSV_PATH, index=False)

        print("ENCODED", flush=True)
