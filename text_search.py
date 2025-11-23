import torch
import clip
import faiss
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = pd.read_csv("data\images_with_captions_and_tags.csv")
image_paths = data["image_path"].tolist()

image_embeddings = np.load("embeddings\image_embeddings_normalized.npy").astype(np.float32)
dim = image_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(image_embeddings)

def search_images_basic(query, k=10):
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        q_emb = model.encode_text(tokens)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

    q_vec = q_emb.cpu().numpy().astype(np.float32)

    scores, indices = index.search(q_vec, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append((image_paths[idx], float(score)))

    return results

if __name__ == "__main__":
    while True:
        query = input("Enter your search query: ")
        if not query.strip():
            query = "a car"
        if query.lower() in ["exit", "quit"]:
            break

        results = search_images_basic(query, k=10)
        print("\nTop Results (pure CLIP):\n")
        for p, s in results:
            print(f"{p}  |  score={s:.4f}")
        print()
        print("-" * 40)
