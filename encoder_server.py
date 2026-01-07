import json
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

df = pd.read_csv(CSV_PATH)
image_paths = df["image_path"].tolist()
image_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

index = faiss.read_index(str(FAISS_INDEX_PATH))


def encode_images(paths: list[str]) -> np.ndarray:
    return embedder.embed_images(paths)


def save_state():
    df.to_csv(CSV_PATH, index=False)
    np.save(EMBEDDINGS_PATH, image_embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))


def rebuild_faiss(embeddings: np.ndarray):
    if len(embeddings) == 0:
        raise RuntimeError("Cannot rebuild FAISS with empty embeddings")

    dim = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(embeddings)
    return new_index


def delete_rows_by_paths(paths: list[str]) -> list[int]:
    global df

    mask = df["image_path"].isin(paths)
    deleted_idx = df.index[mask].to_list()

    if deleted_idx:
        df = df[~mask].reset_index(drop=True)

    return deleted_idx


print("ENCODER_READY", flush=True)

while True:
    cmd = sys.stdin.readline().strip()

    if not cmd:
        continue

    if cmd == "EXIT":
        break

    if cmd == "UPDATE_JSON":
        new_paths = json.loads(sys.stdin.readline())

        if not new_paths:
            print("ERROR Empty update payload", flush=True)
            continue

        new_embeddings = encode_images(new_paths)

        if new_embeddings.shape[1] != index.d:
            print("ERROR Embedding dimension mismatch", flush=True)
            continue

        index.add(new_embeddings)

        image_embeddings = np.vstack([image_embeddings, new_embeddings])
        image_paths.extend(new_paths)

        df = pd.DataFrame({"image_path": image_paths})

        save_state()
        print("ENCODED", flush=True)
        continue

    if cmd == "DELETE":
        paths_to_delete = json.loads(sys.stdin.readline())

        if not paths_to_delete:
            print("ERROR Empty delete payload", flush=True)
            continue

        deleted_idx = delete_rows_by_paths(paths_to_delete)

        if not deleted_idx:
            print("DELETED 0", flush=True)
            continue

        image_embeddings = np.delete(image_embeddings, deleted_idx, axis=0)
        index = rebuild_faiss(image_embeddings)

        image_paths = df["image_path"].tolist()

        save_state()
        print(f"DELETED {len(deleted_idx)}", flush=True)
        continue

    print(f"ERROR Unknown command: {cmd}", flush=True)
