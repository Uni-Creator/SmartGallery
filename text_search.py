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

def search_images_basic(query, limit=100, alpha=0.23):
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        q_emb = model.encode_text(tokens)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

    q_vec = q_emb.cpu().numpy().astype(np.float32)

    raw_k = min(len(image_embeddings), max(limit * 5, 100))

    scores, indices = index.search(q_vec, raw_k)

    results = [
        (image_paths[idx], float(score))
        for idx, score in zip(indices[0], scores[0])
        if score >= alpha
    ]

    # Keep only top "limit" after filtering
    return results[:limit] if limit > len(results) else results


def search_all_above_alpha(query, alpha=0.23):
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        q_emb = model.encode_text(tokens)
        q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

    q_vec = q_emb.cpu().numpy().astype(np.float32)

    # Exact cosine similarity (dot product because embeddings are normalized)
    sims = image_embeddings @ q_vec.T
    sims = sims.squeeze()

    results = [
        (image_paths[i], float(sim))
        for i, sim in enumerate(sims)
        if sim >= alpha
    ]

    # Sort descending by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    return results


# if __name__ == "__main__":
    
    # try:
    #     import sys
    #     query = sys.argv[1]
        
    #     results = search_images_basic(query)
    #     print("\nTop Results (pure CLIP):\n")
    #     for p, s in results:
    #         print(f"{p}  |  score={s:.4f}")
    #     print()
        
    # except IndexError:        
    #     while True:
    #         query = input("Enter your search query: ")
    #         if not query.strip():
    #             query = "a cat"
    #         if query.lower() in ["exit", "quit"]:
    #             break

    #         results = search_images_basic(query)
    #         print("\nTop Results (pure CLIP):\n")
    #         for p, s in results:
    #             print(f"'{p}'  |  score={s:.4f}")
    #         print()
    #         print("-" * 40)
    
results = search_all_above_alpha("birthday", alpha=0.23)
print(len(results), "results found:")
# for path, score in results:
#     print(path, score)
