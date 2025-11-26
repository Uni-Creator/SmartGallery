import os
import pandas as pd
from pathlib import Path


# Step 1: Configuration parameters and paths

BASE_DIR = Path(r"D:\Personal\PHOTOS")
VALID_EXTENSIONS = {"jpg", "jpeg", "png"}

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "final_cleaned_data.csv"


# Step 2: Scan directory for image files

def scan_images(base_dir,  valid_extensions):

    print("Scanning directory for image files...")
    image_paths = []

    for root, dirs, files in os.walk(base_dir):

        for file in files:
            if file.lower().split(".")[-1] in valid_extensions:
                image_paths.append(Path(root) / file)

    print(f"Total files found: {len(image_paths)}")

    image_ids = [f"img_{i+1:05d}" for i in range(len(image_paths))]

    df = pd.DataFrame({
        "image_id": image_ids,
        "image_path": [str(p) for p in image_paths]
    })

    return df


# Step 3: Extract group, subgroup and event from image path

def extract_group_info(path):

    parts = Path(path).parts

    if "PHOTOS" in parts:
        base_index = parts.index("PHOTOS")

        group = parts[base_index + 1] if len(parts) > base_index + 1 else "Unknown"
        subgroup = parts[base_index + 2] if len(parts) > base_index + 2 else "Unknown"
        event = parts[base_index + 3] if len(parts) > base_index + 3 else ""
    else:
        group = "Unknown"
        subgroup = "Unknown"
        event = ""

    return group, subgroup, event


# Step 4: Clean event and subgroup values

def clean_event(event):

    if pd.isna(event):
        return "no_event"

    event = str(event).strip().lower()

    if event.endswith((".jpg", ".jpeg", ".png")) or event == "":
        return "no_event"

    return event


def clean_subgroup(subgroup):

    if pd.isna(subgroup):
        return "no_subgroup"

    subgroup = str(subgroup).strip().lower()

    if subgroup.endswith((".jpg", ".jpeg", ".png")) or subgroup == "":
        return "no_subgroup"

    return subgroup


# Step 5: Build final dataframe

def main():

    df = scan_images(BASE_DIR, VALID_EXTENSIONS)

    print("Extracting group, subgroup, and event name...")
    group_data = df["image_path"].apply(extract_group_info)

    df["group"] = group_data.map(lambda x: x[0])
    df["subgroup"] = group_data.map(lambda x: x[1])
    df["event_name"] = group_data.map(lambda x: x[2])

    print("Cleaning subgroup and event...")
    df["subgroup"] = df["subgroup"].apply(clean_subgroup)
    df["event_name"] = df["event_name"].apply(clean_event)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Final cleaned dataset saved to: {OUTPUT_FILE}")


# Step 6: Script entry point

if __name__ == "__main__":
    main()
