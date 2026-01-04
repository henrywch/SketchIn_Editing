"""
Used to download FSCOCO Datasets
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
# 1. Set the Root path (The folder containing images/, raster_sketches/, text/)
DATASET_ROOT = Path("/inspire/hdd/project/video-understanding/public/share/datasets/fscoco/fscoco")

# 2. Define the sub-folders
# We use 'raster_sketches' because these are the Human Scene Sketches.
IMG_DIR_NAME = "images"
SKETCH_DIR_NAME = "raster_sketches" 
TXT_DIR_NAME = "text"

# 3. Output file
OUTPUT_JSONL = DATASET_ROOT / "metadata.jsonl"
# =====================

def scan_directory(directory: Path, extensions: set):
    """
    Recursively finds all files with given extensions.
    Returns a dictionary: { "filename_stem": "full_absolute_path" }
    """
    print(f"Scanning {directory}...")
    file_map = {}
    
    # rglob('*') recursively finds all files
    for path in tqdm(directory.rglob('*'), desc=f"Indexing {directory.name}"):
        if path.is_file() and path.suffix.lower() in extensions:
            # The 'stem' is '00000018485' (unique COCO ID)
            stem = path.stem
            
            # Safety check for duplicates (optional)
            if stem in file_map:
                # If duplicates exist (e.g. in train and val folders), 
                # we usually keep the first one or warn. 
                # For COCO IDs, they should be unique across the whole dataset usually.
                pass 
            else:
                file_map[stem] = str(path.absolute())
                
    return file_map

def build_metadata():
    # 1. Index Images (Target)
    # Images can be .jpg or .png
    img_map = scan_directory(DATASET_ROOT / IMG_DIR_NAME, {'.jpg', '.jpeg', '.png'})
    
    # 2. Index Sketches (Condition)
    # Sketches are usually .png
    sketch_map = scan_directory(DATASET_ROOT / SKETCH_DIR_NAME, {'.png', '.jpg', '.jpeg'})
    
    # 3. Index Text (Prompt)
    txt_map = scan_directory(DATASET_ROOT / TXT_DIR_NAME, {'.txt'})
    
    print(f"\nFound {len(img_map)} images.")
    print(f"Found {len(sketch_map)} sketches.")
    print(f"Found {len(txt_map)} caption files.")

    # 4. Intersect and Write
    valid_count = 0
    missing_sketch_count = 0
    missing_text_count = 0
    
    print(f"\nMatching pairs and writing to {OUTPUT_JSONL}...")
    
    with open(OUTPUT_JSONL, 'w') as f_out:
        for stem, img_path in tqdm(img_map.items(), desc="Writing JSONL"):
            
            # Find matching Image
            sketch_path = sketch_map.get(stem)
            if not sketch_path:
                missing_sketch_count += 1
                
                continue
            
            # Find matching Text
            txt_path = txt_map.get(stem)
            caption_text = ""
            
            if txt_path:
                try:
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        caption_text = f.read().strip()
                except Exception as e:
                    print(f"Error reading text {txt_path}: {e}")
            else:
                missing_text_count += 1
                
                continue
                
            # If we are here, we have at least Image + Sketch
            entry = {
                "image_path": img_path,
                "conditioning_path": sketch_path,
                "caption": caption_text
            }
            
            f_out.write(json.dumps(entry) + "\n")
            valid_count += 1

    print("\n=== Summary ===")
    print(f"Total pairs created: {valid_count}")
    print(f"Images with missing sketch: {missing_sketch_count}")
    print(f"Images with missing text: {missing_text_count}")
    print(f"Output saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    if not (DATASET_ROOT / IMG_DIR_NAME).exists():
        print(f"Error: Image directory not found at {DATASET_ROOT / IMG_DIR_NAME}")
    elif not (DATASET_ROOT / SKETCH_DIR_NAME).exists():
        print(f"Error: Sketch directory not found at {DATASET_ROOT / SKETCH_DIR_NAME}")
    elif not (DATASET_ROOT / TXT_DIR_NAME).exists():
        print(f"Error: Text directory not found at {DATASET_ROOT / TXT_DIR_NAME}")
    else:
        build_metadata()