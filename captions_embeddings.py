import torch
import clip
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = pd.read_csv("data\images_with_captions_and_tags.csv")
captions = data["image_caption"].astype(str).tolist()

text_embeddings = []

with torch.no_grad():
    for cap in captions:
        text = clip.tokenize([cap]).to(device)
        emb = model.encode_text(text)
        text_embeddings.append(emb.cpu().numpy()[0])

import numpy as np
np.save("embeddings\caption_embeddings.npy", text_embeddings)

print("Caption embeddings saved.")
