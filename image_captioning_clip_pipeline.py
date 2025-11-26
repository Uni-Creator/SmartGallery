import numpy as np
import torch
import pandas as pd
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from tqdm import tqdm
from keybert import KeyBERT
import clip
from pathlib import Path


# Step 1: Configuration parameters and paths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")

BATCH_SIZE = 8
MAX_LENGTH = 16
NUM_BEAMS = 4

DATA_DIR = Path("data")
EMBED_DIR = Path("embeddings")

DATA_DIR.mkdir(exist_ok=True)
EMBED_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "final_cleaned_data.csv"
OUT_CSV_PATH = DATA_DIR / "images_with_captions_and_tags.csv"
CAPTION_EMBED_PATH = EMBED_DIR / "caption_embeddings.npy"
IMAGE_EMBED_PATH = EMBED_DIR / "image_embeddings.npy"


# Step 2: Load all required models

def load_models(device):

    # Caption generation model
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    caption_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    caption_model.eval()

    # Tag extraction model
    kw_model = KeyBERT()

    # CLIP model for embeddings
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    return caption_model, processor, tokenizer, kw_model, clip_model, clip_preprocess


# Step 3: Utility function for safe image loading

def load_image_safe(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARNING] Image load failed: {path} | {e}")
        return Image.new("RGB", (224, 224))


# Step 4: Batch caption generation

def generate_captions_batch(image_paths, caption_model, processor, tokenizer, device):
    images = [load_image_safe(p) for p in image_paths]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(
            inputs.pixel_values,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS
        )

    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]


# Step 5: Tag extraction using KeyBERT

def extract_tags(caption, kw_model):
    keywords = kw_model.extract_keywords(
        caption,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )
    return [kw[0] for kw in keywords]


# Step 6: Generate captions and tags for all images

def generate_captions_and_tags(image_paths, caption_model, processor, tokenizer, kw_model, device):
    captions = []
    tags = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i + BATCH_SIZE]

        batch_captions = generate_captions_batch(
            batch_paths,
            caption_model,
            processor,
            tokenizer,
            device
        )

        for caption in batch_captions:
            captions.append(caption)
            tags.append(", ".join(extract_tags(caption, kw_model)))

    return captions, tags


# Step 7: Generate CLIP embeddings for captions

def embed_captions(captions, clip_model, device, output_path):
    embeddings = []

    with torch.no_grad():
        for caption in tqdm(captions):
            tokens = clip.tokenize([caption]).to(device)
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy()[0])

    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(output_path, embeddings)

    return embeddings


# Step 8: Generate CLIP embeddings for images

def embed_images(image_paths, clip_model, preprocess, device, output_path):
    embeddings = []

    with torch.no_grad():
        for path in tqdm(image_paths):
            try:
                image = Image.open(path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                emb = clip_model.encode_image(image_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)

                embeddings.append(emb.cpu().numpy()[0])

            except Exception as e:
                print(f"[WARNING] Image embedding failed: {path} | {e}")
                embeddings.append(np.zeros(512, dtype=np.float32))

    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(output_path, embeddings)

    return embeddings


# Step 9: Main pipeline execution

def main():

    caption_model, processor, tokenizer, kw_model, clip_model, clip_preprocess = load_models(DEVICE)

    # Step 9.1: Load dataset
    df = pd.read_csv(DATA_PATH)
    image_paths = df["image_path"].astype(str).tolist()

    # Step 9.2: Generate captions and tags
    print("Generating captions and tags...")
    captions, tags = generate_captions_and_tags(
        image_paths,
        caption_model,
        processor,
        tokenizer,
        kw_model,
        DEVICE
    )

    df["image_caption"] = captions
    df["image_tags"] = tags

    df.to_csv(OUT_CSV_PATH, index=False)
    print(f"Saved captions and tags to: {OUT_CSV_PATH}")

    # Step 9.3: Generate caption embeddings
    user_choice = input("Generate caption embeddings using CLIP? (y/n): ").strip().lower()
    if user_choice.lower() == "y":
        embed_captions(df["image_caption"].astype(str).tolist(), clip_model, DEVICE, CAPTION_EMBED_PATH)
        print(f"Caption embeddings saved to: {CAPTION_EMBED_PATH}")

    # Step 9.4: Generate image embeddings
    user_choice = input("Generate image embeddings using CLIP? (y/n): ").strip().lower()
    if user_choice.lower() == "y":
        embed_images(image_paths, clip_model, clip_preprocess, DEVICE, IMAGE_EMBED_PATH)
        print(f"Image embeddings saved to: {IMAGE_EMBED_PATH}")


# Step 10: Entry point

if __name__ == "__main__":
    main()
