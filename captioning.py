from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
from keybert import KeyBERT

# ================== MODELS ==================

# Caption model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tag extraction model
kw_model = KeyBERT()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()

# ================== SETTINGS ==================

BATCH_SIZE = 8
max_length = 16
num_beams = 4

# ================== FUNCTIONS ==================

def generate_captions_batch(image_paths):
    images = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except:
            images.append(Image.new("RGB", (224, 224)))

    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=max_length,
            num_beams=num_beams
        )

    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [c.strip() for c in captions]


def extract_tags(caption):
    keywords = kw_model.extract_keywords(
        caption,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=5
    )
    tags = [kw[0] for kw in keywords]
    return tags


# ================== PIPELINE ==================

data = pd.read_csv("data\\final_grouped_images.csv")
image_paths = data["image_path"].tolist()

captions = []
tags_list = []

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_captions = generate_captions_batch(batch_paths)

    for cap in batch_captions:
        captions.append(cap)
        tags_list.append(", ".join(extract_tags(cap)))

# ================== SAVE RESULT ==================

data["image_caption"] = captions
data["image_tags"] = tags_list

data.to_csv("data\images_with_captions_and_tags.csv", index=False)

print("Captions + tags generated and saved.")
