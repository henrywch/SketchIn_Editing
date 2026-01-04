"""
Pair the Filtered Danbooru Results
"""

import json
import argparse
import os
from pathlib import Path

def extract_id_and_type(entry):
    """
    Parses the image_path to determine ID and Type (sketch vs src).
    Returns: (id_str, type_str) or (None, None)
    """
    path_str = entry.get("image_path", "")
    
    # 1. Determine Type based on folder structure
    # Adjust string checks if your folder naming varies slightly
    if "/sketch/" in path_str:
        img_type = "sketch"
    elif "/src/" in path_str:
        img_type = "src"
    else:
        return None, None

    # 2. Extract ID from filename
    # Example: ".../1012000.png" -> "1012000"
    try:
        filename = os.path.basename(path_str)
        img_id = os.path.splitext(filename)[0]
        return img_id, img_type
    except Exception:
        return None, None

def pair_datasets(input_file, output_paired, output_unpaired):
    print(f"Reading {input_file}...")
    
    # Dictionary to hold data: { "id": entry_dict }
    sketches_map = {}
    images_map = {}
    
    # 1. Read and Partition
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                img_id, img_type = extract_id_and_type(entry)
                
                if img_id:
                    if img_type == "sketch":
                        sketches_map[img_id] = entry
                    elif img_type == "src":
                        images_map[img_id] = entry
            except:
                continue

    print(f"Found {len(sketches_map)} potential sketches.")
    print(f"Found {len(images_map)} potential source images.")

    # 2. Find Intersection (Pairs)
    sketch_ids = set(sketches_map.keys())
    image_ids = set(images_map.keys())
    
    common_ids = sketch_ids.intersection(image_ids)
    unpaired_sketch_ids = sketch_ids - image_ids
    unpaired_image_ids = image_ids - sketch_ids

    # 3. Create Paired Output
    print(f"Pairing {len(common_ids)} entries...")
    
    with open(output_paired, 'w', encoding='utf-8') as f_out:
        for uid in common_ids:
            sketch_entry = sketches_map[uid]
            src_entry = images_map[uid]
            
            # Construct merged entry
            merged = {
                "id": uid,
                "image": src_entry["image_path"],
                "sketch": sketch_entry["image_path"],
            }
            
            # Add Scores (Prefixing to avoid collision)
            # e.g., 'musiq' becomes 'image_musiq' and 'sketch_musiq'
            for key, val in src_entry.items():
                if key != "image_path" and key != "idx":
                    merged[f"image_{key}"] = val
            
            for key, val in sketch_entry.items():
                if key != "image_path" and key != "idx":
                    merged[f"sketch_{key}"] = val
            
            f_out.write(json.dumps(merged) + "\n")

    # 4. Create Unpaired Output (Leftovers)
    count_unpaired = 0
    with open(output_unpaired, 'w', encoding='utf-8') as f_out:
        # Write leftover sketches
        for uid in unpaired_sketch_ids:
            f_out.write(json.dumps(sketches_map[uid]) + "\n")
            count_unpaired += 1
            
        # Write leftover images
        for uid in unpaired_image_ids:
            f_out.write(json.dumps(images_map[uid]) + "\n")
            count_unpaired += 1

    print("-" * 30)
    print(f"DONE.")
    print(f"Successfully Paired: {len(common_ids)}")
    print(f"Unpaired Leftovers:  {count_unpaired}")
    print(f"Saved pairs to:      {output_paired}")
    print(f"Saved leftovers to:  {output_unpaired}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the filtered scores jsonl")
    parser.add_argument("--output_paired", type=str, default="final_paired.jsonl")
    parser.add_argument("--output_unpaired", type=str, default="leftovers.jsonl")
    args = parser.parse_args()

    if os.path.exists(args.input):
        pair_datasets(args.input, args.output_paired, args.output_unpaired)
    else:
        print("Input file not found.")