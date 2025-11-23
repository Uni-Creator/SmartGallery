import faiss
import numpy as np

embeddings = np.load("embeddings\image_embeddings_normalized.npy")

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)  
index.add(embeddings)

print("FAISS index built with", index.ntotal, "images")
faiss.write_index(index, "embeddings\image_faiss_index.idx")