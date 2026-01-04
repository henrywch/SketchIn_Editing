"""
Used to download SketchyScene Datasets
"""

import os
import json
import re
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
DATASET_ROOT = Path("/inspire/hdd/project/video-understanding/public/share/datasets/SketchyScene/SketchyScene-7k/train")

# Folder Names matching your screenshots
DIRS = {
    "image": "reference_image",
    "sketch": "DRAWING_GT",
    "instance": "INSTANCE_GT",
    "class": "CLASS_GT"
}

OUTPUT_JSONL = DATASET_ROOT / "metadata.jsonl"
# =====================

def extract_id(filename):
    """
    Robust ID extraction for SketchyScene variants:
    - '1629.jpg' -> '1629'
    - 'L0_sample1629.png' -> '1629'
    - 'sample_1629_instance.mat' -> '1629'
    - 'sample_1629_class.mat' -> '1629'
    """
    # Find all sequences of digits
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # The ID is typically the last number in the sequence 
        # (e.g. sample_1629 -> 1629)
        return numbers[-1]
    return None

def scan_folder(folder_path, extensions):
    """Returns a dict { 'ID': 'absolute_path' }"""
    if not folder_path.exists():
        print(f"Warning: Folder not found {folder_path}")
        return {}
        
    mapping = {}
    # Scan for files with given extensions
    for ext in extensions:
        for path in folder_path.glob(f"*{ext}"):
            file_id = extract_id(path.name)
            if file_id:
                mapping[file_id] = str(path.absolute())
    return mapping

def build_metadata():
    print("Scanning directories...")
    
    # 1. Index everything
    img_map = scan_folder(DATASET_ROOT / DIRS["image"], {".jpg", ".png"})
    sketch_map = scan_folder(DATASET_ROOT / DIRS["sketch"], {".png", ".jpg"})
    inst_map = scan_folder(DATASET_ROOT / DIRS["instance"], {".mat"})
    class_map = scan_folder(DATASET_ROOT / DIRS["class"], {".mat"})

    print(f"Found: {len(img_map)} imgs, {len(sketch_map)} sketches, {len(inst_map)} inst-mats, {len(class_map)} class-mats")

    # 2. Match pairs
    valid_entries = []
    print("Linking files by ID...")
    
    # We use images as the anchor
    for file_id, img_path in tqdm(img_map.items()):
        # We require at least Image + Sketch
        if file_id in sketch_map:
            entry = {
                "id": file_id,
                "image_path": img_path,
                "conditioning_path": sketch_map[file_id],
                "caption": "A cartoon sketch of a scene", # Default caption
                
                # Optional fields (might be None if missing)
                "instance_path": inst_map.get(file_id, None),
                "class_path": class_map.get(file_id, None)
            }
            valid_entries.append(entry)

    # 3. Write to JSONL
    print(f"Writing {len(valid_entries)} complete entries to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, 'w') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry) + "\n")

    print("Done!")

if __name__ == "__main__":
    build_metadata()