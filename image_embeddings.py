import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = pd.read_csv("data\images_with_captions_and_tags.csv")
image_paths = data["image_path"].tolist()

embeddings = []

with torch.no_grad():
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
        except:
            embeddings.append(np.zeros(512, dtype=np.float32))
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)
        emb = model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # IMPORTANT

        embeddings.append(emb.cpu().numpy()[0].astype(np.float32))

embeddings = np.array(embeddings, dtype=np.float32)
np.save("embeddings\image_embeddings_normalized.npy", embeddings)

print("Saved normalized image embeddings to image_embeddings_normalized.npy")
