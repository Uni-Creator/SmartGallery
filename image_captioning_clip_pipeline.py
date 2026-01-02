import numpy as np
import torch
import pandas as pd
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from tqdm import tqdm
from keybert import KeyBERT
import clip
from pathlib import Path
import logging
from typing import List


# Step 1: Configuration parameters and paths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

BATCH_SIZE = 8
MAX_LENGTH = 16
NUM_BEAMS = 4
CLIP_BATCH_SIZE = 32

DATA_DIR = Path("data")
EMBED_DIR = Path("embeddings")

DATA_DIR.mkdir(exist_ok=True)
EMBED_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "images_path.csv"
OUT_CSV_PATH = DATA_DIR / "images_with_captions_and_tags.csv"
CAPTION_EMBED_PATH = EMBED_DIR / "caption_embeddings.npy"
IMAGE_EMBED_PATH = EMBED_DIR / "image_embeddings.npy"


class EmbeddingGenerator:

    def __init__(self, device: torch.device):
        self.device = device
        self._load_models()

    # Step 2: Load all required models

    def _load_models(self):

        model_name = "nlpconnect/vit-gpt2-image-captioning"

        self.caption_model = (
            VisionEncoderDecoderModel
            .from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.kw_model = KeyBERT()

        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=self.device
        )
        self.clip_model.eval()

    # Step 3: Utility function for safe image loading

    @staticmethod
    def load_image_safe(path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARNING] Image load failed: {path} | {e}")
            return Image.new("RGB", (224, 224))

    # Step 4: Batch caption generation

    def generate_captions_batch(self, image_paths: List[str]) -> List[str]:
        images = [self.load_image_safe(p) for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.caption_model.generate(
                inputs.pixel_values,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
            )

        captions = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return [c.strip() for c in captions]

    # Step 5: Tag extraction using KeyBERT

    def extract_tags(self, caption: str) -> List[str]:
        keywords = self.kw_model.extract_keywords(
            caption,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5,
        )
        return [kw[0] for kw in keywords]

    # Step 6: Generate captions and tags for all images

    def generate_captions_and_tags(self, image_paths: List[str]):
        captions, tags = [], []

        for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            batch_captions = self.generate_captions_batch(batch_paths)

            for caption in batch_captions:
                captions.append(caption)
                tags.append(", ".join(self.extract_tags(caption)))

        return captions, tags

    # Step 7: Generate CLIP embeddings for captions

    def embed_captions(self, captions: List[str]):
        embeddings = []

        for i in tqdm(range(0, len(captions), CLIP_BATCH_SIZE)):
            batch = captions[i:i + CLIP_BATCH_SIZE]
            tokens = clip.tokenize(batch).to(self.device)

            with torch.no_grad():
                emb = self.clip_model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.cpu().numpy())

        embeddings = np.vstack(embeddings).astype(np.float32)
        return embeddings

    # Step 8: Generate CLIP embeddings for images

    def embed_images(self, image_paths: List[str]):
        embeddings = []

        for i in tqdm(range(0, len(image_paths), CLIP_BATCH_SIZE)):
            batch_paths = image_paths[i:i + CLIP_BATCH_SIZE]

            images = [
                self.clip_preprocess(self.load_image_safe(p))
                for p in batch_paths
            ]
            image_tensor = torch.stack(images).to(self.device)

            with torch.no_grad():
                emb = self.clip_model.encode_image(image_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.cpu().numpy())

        embeddings = np.vstack(embeddings).astype(np.float32)
        return embeddings


# Step 9: Main pipeline execution

def main():

    embd_gen = EmbeddingGenerator(DEVICE)

    # Step 9.1: Load dataset
    df = pd.read_csv(DATA_PATH)
    image_paths = df["image_path"].astype(str).tolist()

    # Step 9.2: Generate captions and tags
    print("Generating captions and tags...")
    captions, tags = embd_gen.generate_captions_and_tags(image_paths)

    df["image_caption"] = captions
    df["image_tags"] = tags
    df.to_csv(OUT_CSV_PATH, index=False)

    print(f"Saved captions and tags to: {OUT_CSV_PATH}")

    # Step 9.3: Generate caption embeddings
    user_choice = input("Generate caption embeddings using CLIP? (y/n): ").strip().lower()
    if user_choice == "y":
        embd_gen.embed_captions(captions, CAPTION_EMBED_PATH)
        np.save(CAPTION_EMBED_PATH, embeddings)
        print(f"Caption embeddings saved to: {CAPTION_EMBED_PATH}")

    # Step 9.4: Generate image embeddings
    user_choice = input("Generate image embeddings using CLIP? (y/n): ").strip().lower()
    if user_choice == "y":
        embeddings = embd_gen.embed_images(image_paths)
        np.save(IMAGE_EMBED_PATH, embeddings)
        print(f"Image embeddings saved to: {IMAGE_EMBED_PATH}")


# Step 10: Entry point

if __name__ == "__main__":
    main()
