"""
Sample the FSCOCO Datasets
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
# Path to the BIG metadata file you created in the previous step
SOURCE_METADATA_PATH = "/inspire/hdd/project/video-understanding/public/share/datasets/fscoco/fscoco/metadata.jsonl"

# Name of the new small metadata file
OUTPUT_METADATA_NAME = "metadata.jsonl"

# Categories to sample (1 to 100)
TARGET_CATEGORIES = [str(i) for i in range(1, 101)]
SAMPLES_PER_CATEGORY = 10
# =================================================

def sample_dataset():
    source_path = Path(SOURCE_METADATA_PATH)
    if not source_path.exists():
        print(f"Error: Could not find source metadata at {source_path}")
        return

    print("1. Reading source metadata and grouping by category...")
    
    # Dictionary to hold lists of items: { "69": [item1, item2...], "70": [...] }
    grouped_data = {}

    with open(source_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                item = json.loads(line)
                
                # Extract category ID from the path. 
                # Assuming path ends in: .../images/CATEGORY/FILENAME.jpg
                # We use pathlib to get the parent folder name.
                img_path = Path(item['image_path'])
                category_id = img_path.parent.name
                
                # Only store if it's strictly in our target range "1" to "100"
                # (Or strictly numeric, depending on your folder structure)
                if category_id in TARGET_CATEGORIES:
                    if category_id not in grouped_data:
                        grouped_data[category_id] = []
                    grouped_data[category_id].append(item)
            except Exception as e:
                print(f"Skipping line due to error: {e}")

    print(f"Found data for {len(grouped_data)} categories within the range 1-100.")

    # List to store the new metadata entries
    new_metadata_entries = []
    
    print("2. Copying files and generating new metadata...")
    
    # Iterate through our target categories 1 to 100
    for cat_id in tqdm(TARGET_CATEGORIES):
        if cat_id not in grouped_data:
            continue
            
        # Get the first 10 items (or fewer if less than 10 exist)
        items_to_process = grouped_data[cat_id][:SAMPLES_PER_CATEGORY]
        
        for item in items_to_process:
            src_img_path = Path(item['image_path'])
            src_cond_path = Path(item['conditioning_path'])
            
            # Define new relative paths (e.g., images/69/0000494112.jpg)
            # We enforce forward slashes for the JSON string to match request
            dest_img_rel = f"images/{cat_id}/{src_img_path.name}"
            dest_cond_rel = f"raster_sketches/{cat_id}/{src_cond_path.name}"
            
            # Define absolute destination paths for copying
            dest_img_abs = Path.cwd() / "images" / cat_id / src_img_path.name
            dest_cond_abs = Path.cwd() / "raster_sketches" / cat_id / src_cond_path.name
            
            try:
                # 1. Create subdirectories
                dest_img_abs.parent.mkdir(parents=True, exist_ok=True)
                dest_cond_abs.parent.mkdir(parents=True, exist_ok=True)
                
                # 2. Copy files
                shutil.copy2(src_img_path, dest_img_abs)
                shutil.copy2(src_cond_path, dest_cond_abs)
                
                # 3. Add to new metadata list
                new_entry = {
                    "image_path": dest_img_rel,
                    "conditioning_path": dest_cond_rel,
                    "caption": item['caption']
                }
                new_metadata_entries.append(new_entry)
                
            except FileNotFoundError:
                print(f"Warning: Source file missing for {src_img_path.name}")
            except Exception as e:
                print(f"Error copying {src_img_path.name}: {e}")

    # 3. Save the new JSONL file
    output_path = Path.cwd() / OUTPUT_METADATA_NAME
    print(f"3. Saving new metadata to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in new_metadata_entries:
            f.write(json.dumps(entry) + "\n")

    print("Done! Sampled dataset created.")

if __name__ == "__main__":
    sample_dataset()