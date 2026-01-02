import sys
from pathlib import Path
from text_search import CLIPSearchEngine

# Step 1: Initialize search engine

CSV_PATH = Path("data") / "images_path.csv"
EMBED_PATH = Path("embeddings") / "image_embeddings_normalized.npy"

engine = CLIPSearchEngine(CSV_PATH, EMBED_PATH)

print("READY", flush=True)

# Step 2: Read queries from stdin loop

while True:
    query = sys.stdin.readline().strip()

    if query == "EXIT":
        break
    # results = engine.search_top_k(query, k=50, alpha=0.23)
    results = engine.search_all_above_alpha(query)

    for path, score in results:
        print(path)

    print("END", flush=True)
