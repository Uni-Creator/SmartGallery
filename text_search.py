import torch
import clip
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import argparse


class CLIPSearchEngine:
    """
    CLIP-based image search engine using FAISS for fast similarity search.

    Loads:
    - Precomputed image embeddings
    - Image file paths (via CSV)
    - CLIP model for query encoding

    Supports:
    - FAISS-based Top-K search
    - Brute force cosine similarity search
    """

    def __init__(self,
                 csv_path: Path,
                 embedding_path: Path,
                 device: str = None):
        """
        Initialize the search engine.

        :param csv_path: Path to CSV containing image metadata and paths
        :param embedding_path: Path to numpy file containing image embeddings
        :param device: Device to run CLIP on ("cuda" or "cpu")
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.csv_path = csv_path
        self.embedding_path = embedding_path

        self._load_clip()
        self._load_data()
        self._build_faiss_index()

    def _load_clip(self):
        """
        Load the CLIP model and preprocessing pipeline.
        """
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def _load_data(self):
        """
        Load image paths and image embeddings.

        Raises:
            FileNotFoundError: If CSV or embedding file does not exist.
            ValueError: If number of image paths and embeddings mismatch.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")

        self.data = pd.read_csv(self.csv_path)
        self.image_paths = self.data["image_path"].tolist()

        self.image_embeddings = np.load(self.embedding_path).astype(np.float32)

        if len(self.image_paths) != len(self.image_embeddings):
            raise ValueError("Mismatch between number of paths and embeddings")

    def _build_faiss_index(self):
        """
        Build FAISS index using inner product (cosine similarity on normalized vectors).
        """
        dim = self.image_embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.image_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a text query into a normalized CLIP embedding.

        :param query: Input textual query
        :return: Normalized embedding vector of shape (1, dim)
        """
        with torch.no_grad():
            tokens = clip.tokenize([query]).to(self.device)
            q_emb = self.model.encode_text(tokens)
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

        return q_emb.cpu().numpy().astype(np.float32)

    def search_top_k(self,
                     query: str,
                     k: int = 50,
                     alpha: float = 0.23) -> List[Tuple[str, float]]:
        """
        Search top K images using FAISS index.

        :param query: Text query for image retrieval
        :param k: Number of top results to return
        :param alpha: Minimum similarity threshold
        :return: List of (image_path, similarity_score)
        """
        q_vec = self.encode_query(query)

        raw_k = min(len(self.image_embeddings), max(k * 5, 100))
        scores, indices = self.index.search(q_vec, raw_k)

        results = [
            (self.image_paths[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if score >= alpha
        ]

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k]

    def search_all_above_alpha(self,
                               query: str,
                               alpha: float = 0.23) -> List[Tuple[str, float]]:
        """
        Perform brute-force cosine similarity search.

        Suitable for small datasets (<10k images).

        :param query: Text query for image retrieval
        :param alpha: Minimum similarity threshold
        :return: List of (image_path, similarity_score)
        """
        q_vec = self.encode_query(query)

        sims = self.image_embeddings @ q_vec.T
        sims = sims.squeeze()

        results = [
            (self.image_paths[i], float(sim))
            for i, sim in enumerate(sims)
            if sim >= alpha
        ]

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def main():
    """
    CLI entry point for CLIP-based image search.
    """

    parser = argparse.ArgumentParser(description="CLIP Image Search Engine")
    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--alpha", type=float, default=0.23, help="Similarity threshold")
    parser.add_argument("--k", type=int, default=50, help="Top K results")
    parser.add_argument("--mode", choices=["faiss", "cosine"], default="faiss")

    args = parser.parse_args()

    CSV_PATH = Path("data") / "images_with_captions_and_tags.csv"
    EMBED_PATH = Path("embeddings") / "image_embeddings_normalized.npy"

    engine = CLIPSearchEngine(CSV_PATH, EMBED_PATH)

    if args.query:
        query = args.query
    else:
        query = input("Enter search query: ").strip()

    if args.mode == "faiss":
        results = engine.search_top_k(query, k=args.k, alpha=args.alpha)
    else:
        results = engine.search_all_above_alpha(query, alpha=args.alpha)

    if not results:
        print("No results found above threshold.")
        return

    print("\nSearch Results:\n")
    for path, score in results:
        print(f"{path} | score={score:.4f}")


if __name__ == "__main__":
    main()
