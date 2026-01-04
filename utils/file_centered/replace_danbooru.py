"""
Performing the embedding of  sketch piece on the original image
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_JSONL = "/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/scored.jsonl"
OUTPUT_JSONL = "/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/combined_meta.jsonl"
OUTPUT_BASE_DIR = "/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced"

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def ensure_dirs(base_path):
    """Creates necessary subdirectories."""
    char_dir = os.path.join(base_path, "character_combined")
    head_dir = os.path.join(base_path, "head_combined")
    os.makedirs(char_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)
    return char_dir, head_dir

def get_first_bbox(bbox_dict, key):
    """
    Safely retrieves the first bbox for a key ('character' or 'head').
    Returns (x1, y1, x2, y2) as integers, or None if empty.
    """
    boxes = bbox_dict.get(key, [])
    if not boxes:
        return None
    # Take the first one and convert to int
    return [int(c) for c in boxes[0]]

def perform_replacement(base_img_path, source_sketch_path, base_box, source_box, save_path):
    """
    Crops region from sketch (source_box), resizes to match base_box,
    pastes onto base_img, and saves.
    """
    try:
        # Load images (convert to RGB to avoid palette/alpha issues)
        base_img = Image.open(base_img_path).convert("RGB")
        sketch_img = Image.open(source_sketch_path).convert("RGB")
        
        # 1. Crop the part from the sketch
        # Pillow crop is (left, top, right, bottom)
        sketch_crop = sketch_img.crop(source_box)
        
        # 2. Calculate dimensions of the target area on the base image
        target_w = base_box[2] - base_box[0]
        target_h = base_box[3] - base_box[1]
        
        # 3. Resize sketch crop to fit target area exactly
        # (Handling cases where detection sizes might vary slightly)
        sketch_crop_resized = sketch_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # 4. Paste sketch crop onto base image
        base_img.paste(sketch_crop_resized, (base_box[0], base_box[1]))
        
        # 5. Save
        base_img.save(save_path, quality=95)
        return True
        
    except Exception as e:
        print(f"Error processing {os.path.basename(save_path)}: {e}")
        return False

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------
def process_replacements(input_path, output_path):
    # Setup directories
    char_out_dir, head_out_dir = ensure_dirs(OUTPUT_BASE_DIR)
    
    # Read all lines first to get total count for tqdm
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    new_records = []
    
    # Start loop
    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Generating Combined Images"):
        entry = json.loads(line)
        
        # Original Data
        uid = entry.get("id")
        img_path = entry.get("image")
        sketch_path = entry.get("sketch")
        img_coords = entry.get("img_bbox_coord", {})
        sketch_coords = entry.get("sketch_bbox_coord", {})
        
        # New Record Structure
        new_record = {
            "id": idx,        # New sequential ID (0, 1, 2...)
            "uid": uid,       # Original ID
            "image": img_path,
            "sketch": sketch_path,
            "character_combined": None,
            "head_combined": None
        }
        
        # --- PROCESS CHARACTER ---
        img_char_box = get_first_bbox(img_coords, "character")
        sketch_char_box = get_first_bbox(sketch_coords, "character")
        
        if img_char_box and sketch_char_box:
            save_name = f"{uid}_char_combined.jpg"
            save_full_path = os.path.join(char_out_dir, save_name)
            
            success = perform_replacement(img_path, sketch_path, img_char_box, sketch_char_box, save_full_path)
            if success:
                new_record["character_combined"] = save_full_path

        # --- PROCESS HEAD ---
        img_head_box = get_first_bbox(img_coords, "head")
        sketch_head_box = get_first_bbox(sketch_coords, "head")
        
        if img_head_box and sketch_head_box:
            save_name = f"{uid}_head_combined.jpg"
            save_full_path = os.path.join(head_out_dir, save_name)
            
            success = perform_replacement(img_path, sketch_path, img_head_box, sketch_head_box, save_full_path)
            if success:
                new_record["head_combined"] = save_full_path
                
        new_records.append(new_record)

    # Write Output JSONL
    print(f"Writing metadata to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for record in new_records:
            f_out.write(json.dumps(record) + "\n")
            
    print("Done.")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    process_replacements(INPUT_JSONL, OUTPUT_JSONL)