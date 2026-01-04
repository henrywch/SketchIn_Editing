"""
Re-construct the filtered SketchyScene Datasets
"""

import json
import re
from tqdm import tqdm
import shutil
from pathlib import Path

def restucture_dataset(src_root, target_root, original_jsonl):
    # 1. Setup paths
    src_root = Path(src_root)
    target_root = Path(target_root)
    img_dir = target_root / "images"
    sketch_dir = target_root / "sketches"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    sketch_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load original JSONL into memory for fast lookup
    # mapping: { "id_str": {"caption": "...", "recap": "..."} }
    original_data = {}
    print(f"Loading original metadata from {original_jsonl}...")
    with open(original_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            # Use the 'id' field as the lookup key
            original_data[str(entry.get("id"))] = {
                "caption": entry.get("caption", ""),
                "recap": entry.get("recap", "")
            }

    # 3. Scan source directory for files and extract IDs
    # We use a dict to group image/sketch pairs by ID: { '2468': {'img': path, 'sketch': path} }
    id_pairs = {}
    
    print(f"Scanning {src_root} for files...")
    for file_path in src_root.rglob("*"): # Recursive scan
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        
        # Extract ID from "2468_image_bbox.jpg"
        img_match = re.search(r'(\d+)_image_bbox\.jpg$', filename)
        if img_match:
            file_id = img_match.group(1)
            if file_id not in id_pairs: id_pairs[file_id] = {}
            id_pairs[file_id]['img_src'] = file_path
            continue
            
        # Extract ID from "L0_sample2468_sketch_bbox.png"
        sketch_match = re.search(r'sample(\d+)_sketch_bbox\.png$', filename)
        if sketch_match:
            file_id = sketch_match.group(1)
            if file_id not in id_pairs: id_pairs[file_id] = {}
            id_pairs[file_id]['sketch_src'] = file_path

    # 4. Copy files and generate new metadata
    new_metadata_path = target_root / "metadata.jsonl"
    processed_count = 0

    print(f"Creating new dataset at {target_root}...")
    with open(new_metadata_path, 'w', encoding='utf-8') as f_out:
        for i, (file_id, paths) in enumerate(tqdm(id_pairs.items(), desc="Processing image-sketch pairs")):
            # Only proceed if we found BOTH the image and the sketch, AND it exists in original JSONL
            if 'img_src' in paths and 'sketch_src' in paths:
                if file_id in original_data:
                    # Define new paths
                    new_img_name = f"{file_id}_image.jpg"
                    new_sketch_name = f"{file_id}_sketch.png"
                    
                    target_img_path = img_dir / new_img_name
                    target_sketch_path = sketch_dir / new_sketch_name
                    
                    # Copy files
                    shutil.copy2(paths['img_src'], target_img_path)
                    shutil.copy2(paths['sketch_src'], target_sketch_path)
                    
                    # Create new metadata entry
                    # Using relative paths (standard for datasets)
                    new_entry = {
                        "id": i,
                        "uuid": file_id,
                        "image_path": f"images/{new_img_name}",
                        "sketch_path": f"sketches/{new_sketch_name}",
                        "caption": original_data[file_id]["caption"],
                        "recap": original_data[file_id]["recap"]
                    }
                    
                    f_out.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                    processed_count += 1

    print(f"Done! Processed {processed_count} pairs.")
    print(f"New dataset location: {target_root}")

# --- Configuration ---
SOURCE_DIR = "datasets/bb_ss_filtered"
TARGET_DIR = "datasets/SS_bbox_filtered"
MASTER_JSONL = "datasets/sketchyscene_recaped/recaped.jsonl" # Your original file with captions/recaps

if __name__ == "__main__":
    restucture_dataset(SOURCE_DIR, TARGET_DIR, MASTER_JSONL)