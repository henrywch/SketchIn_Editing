"""
Sample the SketchyScene Datasets
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
# Path to the JSONL created by Script 1
SOURCE_METADATA_PATH = "/inspire/hdd/project/video-understanding/public/share/datasets/SketchyScene/SketchyScene-7k/train/metadata.jsonl"

# Output filename
OUTPUT_METADATA_NAME = "metadata.jsonl"

# How many random samples to take
TOTAL_SAMPLES = 1000 
# =================================================

def sample_dataset():
    source_path = Path(SOURCE_METADATA_PATH)
    if not source_path.exists():
        print(f"Error: Metadata not found at {source_path}")
        return

    print("1. Reading source metadata...")
    all_entries = []
    with open(source_path, 'r') as f:
        for line in f:
            all_entries.append(json.loads(line))
            
    # --- RANDOM SAMPLING ---
    if len(all_entries) > TOTAL_SAMPLES:
        print(f"Randomly selecting {TOTAL_SAMPLES} from {len(all_entries)} total entries...")
        samples = random.sample(all_entries, TOTAL_SAMPLES)
    else:
        print(f"Dataset has fewer than requested samples. Taking all {len(all_entries)}.")
        samples = all_entries

    # Create local directories
    cwd = Path.cwd()
    dirs = {
        "images": cwd / "images",
        "sketches": cwd / "sketches",
        "instance_gt": cwd / "instance_gt",
        "class_gt": cwd / "class_gt"
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True)

    new_metadata_entries = []

    print("2. Copying files...")
    for item in tqdm(samples):
        try:
            new_entry = {}
            
            # --- Helper to copy and record path ---
            def process_file(key, source_path_str, dest_folder_key):
                if source_path_str:
                    src = Path(source_path_str)
                    dest_folder = dirs[dest_folder_key]
                    dest = dest_folder / src.name
                    
                    shutil.copy2(src, dest)
                    
                    # Add relative path to new entry
                    # e.g. "images/1234.jpg"
                    new_entry[key] = f"{dest_folder_key}/{src.name}"
                else:
                    new_entry[key] = None

            # Copy Image
            process_file("image_path", item['image_path'], "images")
            
            # Copy Sketch
            process_file("conditioning_path", item['conditioning_path'], "sketches")
            
            # Copy Instance MAT
            process_file("instance_path", item.get('instance_path'), "instance_gt")
            
            # Copy Class MAT
            process_file("class_path", item.get('class_path'), "class_gt")

            new_metadata_entries.append(new_entry)
            
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")

    # 3. Save New Metadata
    output_path = cwd / OUTPUT_METADATA_NAME
    print(f"3. Saving metadata to {output_path}...")
    
    with open(output_path, 'w') as f:
        for entry in new_metadata_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Done! Data saved to {cwd}")

if __name__ == "__main__":
    sample_dataset()