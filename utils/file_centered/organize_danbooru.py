"""
Organize the Danbooru Inpainting Sketch LoRA Finetuning Scored Results as Metadata
"""

import json
import os
import shutil
import cv2
import numpy as np

def extract_uid_from_path(path):
    """
    Extracts UID from filename like 'lora_13900_sketch_edit_1011041_orig.jpg'.
    Assumes the UID is the second to last element when split by '_'.
    """
    filename = os.path.basename(path)
    # Remove extension
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    # Based on the example: lora_..._edit_1011041_orig
    # The UID '1011041' is the second to last part
    if len(parts) >= 2:
        return parts[-2]
    return None

def draw_and_save_bbox(image_path, bbox_data, output_path):
    """
    Draws the first character bbox on the image and saves it.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found for bbox drawing: {image_path}")
        return

    # Check if 'character' key exists and has at least one bbox
    if 'character' not in bbox_data or not bbox_data['character']:
        return

    # Get the first bbox: [x1, y1, x2, y2]
    bbox = bbox_data['character'][0]
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image: {image_path}")
        return

    # Convert coordinates to int
    x1, y1, x2, y2 = map(int, bbox)

    # Draw rectangle (Color: Red, Thickness: 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imwrite(output_path, img)

def main():
    results_file = '/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/head/results.jsonl'
    scored_file = '/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/scored.jsonl'
    output_jsonl = '/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/head/character_eval.jsonl'
    output_dir = '/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/head/character_filled'

    # 1. Load scored.jsonl into a dictionary for fast lookup by UID
    scored_data = {}
    print(f"Loading {scored_file}...")
    try:
        with open(scored_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Ensure ID is string for consistent matching
                scored_data[str(item['id'])] = item
    except FileNotFoundError:
        print(f"Error: {scored_file} not found.")
        return

    eval_entries = []
    
    # 2. Process results.jsonl
    print(f"Processing {results_file}...")
    try:
        with open(results_file, 'r') as f:
            for line in f:
                res_item = json.loads(line)
                
                # Extract paths
                orig_path = res_item['paths']['orig']
                gen_path = res_item['paths']['gen']
                grid_path = gen_path.replace("gen", "grid")
                
                # Extract UID
                uid = extract_uid_from_path(orig_path)
                if not uid:
                    print(f"Could not extract UID from {orig_path}")
                    continue
                
                # Find matching data in scored.jsonl
                if uid not in scored_data:
                    print(f"UID {uid} not found in {scored_file}")
                    continue
                
                score_item = scored_data[uid]
                
                # Extract required fields from scored
                image_source_path = score_item['image']
                sketch_source_path = score_item['sketch']
                combined_source_path = f"/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/character_combined/{uid}_char_combined.jpg"
                img_bbox_coord = score_item.get('img_bbox_coord', {})
                sketch_bbox_coord = score_item.get('sketch_bbox_coord', {})

                # Check path consistency (as requested)
                if image_source_path != orig_path:
                    # They are likely different paths (dataset vs output dir), but we print for verification
                    # print(f"Note: 'image' path in scored differs from 'orig' in results for UID {uid}")
                    pass

                # 3. Construct new entry for character_eval.jsonl
                new_entry = {
                    "id": res_item['id'],
                    "uid": uid,
                    "image": image_source_path, # Using the original source path
                    "sketch": sketch_source_path,
                    "combined": combined_source_path,
                    "gen": gen_path,
                    "grid": grid_path,
                    "image_bbox_coord": img_bbox_coord,
                    "sketch_bbox_coord": sketch_bbox_coord,
                    "pair_metrics": res_item['pair_metrics'],
                    "original_image_quality": res_item['original_image_quality'],
                    "generated_image_quality": res_item['generated_image_quality']
                }
                eval_entries.append(new_entry)

                # 4. Copy files to subdir
                # Create directory for this UID
                target_subdir = os.path.join(output_dir, uid)
                os.makedirs(target_subdir, exist_ok=True)

                # Define local filenames
                # We use os.path.basename to get filename, or construct specific names
                ext_img = os.path.splitext(image_source_path)[1]
                ext_sketch = os.path.splitext(sketch_source_path)[1]
                ext_gen = os.path.splitext(gen_path)[1]

                local_img_name = f"{uid}_orig{ext_img}"
                local_sketch_name = f"{uid}_sketch{ext_sketch}"
                local_combined_name = f"{uid}_combined{ext_gen}"
                local_gen_name = f"{uid}_gen{ext_gen}"
                local_grid_name = f"{uid}_grid{ext_gen}"

                local_img_path = os.path.join(target_subdir, local_img_name)
                local_sketch_path = os.path.join(target_subdir, local_sketch_name)
                local_combined_path = os.path.join(target_subdir, local_combined_name)
                local_gen_path = os.path.join(target_subdir, local_gen_name)
                local_grid_path = os.path.join(target_subdir, local_grid_name)

                # Helper to copy if file exists
                def copy_file(src, dst):
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        return True
                    else:
                        print(f"Warning: Source file missing: {src}")
                        return False

                # Copy the files
                has_img = copy_file(image_source_path, local_img_path)
                has_sketch = copy_file(sketch_source_path, local_sketch_path)
                copy_file(combined_source_path, local_combined_path)
                copy_file(gen_path, local_gen_path)
                copy_file(grid_path, local_grid_path)

                # 5. Generate BBoxed images
                if has_img:
                    bbox_img_name = f"{uid}_orig_bbox{ext_img}"
                    bbox_img_path = os.path.join(target_subdir, bbox_img_name)
                    draw_and_save_bbox(local_img_path, img_bbox_coord, bbox_img_path)
                
                if has_sketch:
                    bbox_sketch_name = f"{uid}_sketch_bbox{ext_sketch}"
                    bbox_sketch_path = os.path.join(target_subdir, bbox_sketch_name)
                    draw_and_save_bbox(local_sketch_path, sketch_bbox_coord, bbox_sketch_path)

    except FileNotFoundError:
        print(f"Error: {results_file} not found.")
        return

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    # Save the consolidated jsonl
    print(f"Saving {output_jsonl}...")
    with open(output_jsonl, 'w') as f:
        for entry in eval_entries:
            f.write(json.dumps(entry) + '\n')
    
    print("Done.")

if __name__ == "__main__":
    main()