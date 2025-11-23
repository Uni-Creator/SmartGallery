import os
import pandas as pd

# ================== SETTINGS ==================

directory_path = r"D:\Personal\PHOTOS"
exclude_dirs = ["Project", "Snapchat"]
valid_extensions = ["jpeg", "jpg", "png"]

# ================== STEP 1: SCAN IMAGES ==================

print("Scanning directory for image files...")
files_list = []

for root, dirs, files in os.walk(directory_path):

    # Remove excluded directories
    dirs[:] = [d for d in dirs if d not in exclude_dirs]

    for file in files:
        if file.split(".")[-1].lower() in valid_extensions:
            files_list.append(os.path.join(root, file))

print(f"Total files found: {len(files_list)}")

image_ids = [f"img_{i+1:05d}" for i in range(len(files_list))]
df = pd.DataFrame({
    "image_id": image_ids,
    "image_path": files_list
})

# df.to_csv("data\image_files_list.csv", index=False)
# print("Image list CSV created.")


# ================== STEP 2: EXTRACT GROUP INFO ==================

def extract_group_info(path):
    path = path.replace("\\", "/")
    folders = path.split("/")

    if "PHOTOS" in folders:
        base_index = folders.index("PHOTOS")

        group = folders[base_index + 1] if len(folders) > base_index + 1 else "Unknown"
        subgroup = folders[base_index + 2] if len(folders) > base_index + 2 else "Unknown"
        event = folders[base_index + 3] if len(folders) > base_index + 3 else ""

    else:
        group = "Unknown"
        subgroup = "Unknown"
        event = ""

    return group, subgroup, event


groups = df["image_path"].apply(extract_group_info)

df["group"] = [g[0] for g in groups]
df["subgroup"] = [g[1] for g in groups]
df["event_name"] = [g[2] for g in groups]

print("Group, Subgroup, Event extracted.")


# ================== STEP 3: CLEAN EVENT + SUBGROUP ==================

def clean_event(event):
    if pd.isna(event):
        return "no_event"

    event = str(event).strip().lower()

    if event.endswith((".jpg", ".jpeg", ".png")):
        return "no_event"

    if event == "":
        return "no_event"

    return event


def clean_subgroup(subgroup):
    if pd.isna(subgroup):
        return "no_subgroup"

    subgroup = str(subgroup).strip().lower()

    if subgroup.endswith((".jpg", ".jpeg", ".png")):
        return "no_subgroup"

    if subgroup == "":
        return "no_subgroup"

    return subgroup


df["event_name"] = df["event_name"].apply(clean_event)
df["subgroup"] = df["subgroup"].apply(clean_subgroup)

# ================== STEP 4: SAVE FINAL ==================

df.to_csv("data\\final_cleaned_data.csv", index=False)
print("Final cleaned dataset saved as: final_cleaned_data.csv")
