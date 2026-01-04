"""
Simple organizer for the scored finetuning results of danbooru inpainting sketch lora finetuning
"""

import os
import json
from collections import defaultdict

def organize_metadata(folder_path, output_file="metadata.jsonl"):
    """
    Scans a folder for files starting with 'lora_{final_step}_sketch_edit_',
    groups them by their unique ID/Timestamp, and saves them to a JSONL file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Dictionary to hold the groups. 
    # Key = distinct file prefix (group ID), Value = dict of {type: path}
    groups = defaultdict(dict)
    
    # The required specific keys and their corresponding substrings to look for
    # We look for these as suffixes (e.g., _orig) to ensure correct grouping.
    keywords = ["orig", "cond", "gen", "grid"]
    
    # 1. Iterate through files in the folder
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    files = sorted(os.listdir(folder_path))
    
    for filename in files:
        # Filter: Must start with the specific prefix
        if not filename.startswith("lora_5300_sketch_edit_"):
            continue
            
        # Get absolute path
        abs_path = os.path.abspath(os.path.join(folder_path, filename))
        name_no_ext = os.path.splitext(filename)[0]
        
        # 2. Identify which type this file is and group it
        matched = False
        for key in keywords:
            if key in name_no_ext:
                try:
                    # Find the last occurrence of the key to split safely
                    idx = name_no_ext.rfind(key)
                    if idx != -1:
                        # The "group_id" is the string up to the key (stripping trailing chars like "_")
                        group_id = name_no_ext[:idx].rstrip("_")
                        groups[group_id][key] = abs_path
                        matched = True
                        break
                except Exception as e:
                    print(f"Skipping {filename}: {e}")

        if not matched:
            print(f"Warning: Found matching prefix but no known type (orig/cond/gen/grid) in: {filename}")

    # 3. Write to JSONL
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for group_id, record in groups.items():
            if record:
                f.write(json.dumps({'id': count, **record}) + "\n")
                count += 1
                
    print(f"Success! Organized {count} groups into '{output_file}'.")

# --- Usage ---
if __name__ == "__main__":
    # Replace this with your actual image folder path
    TARGET_FOLDER = "/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/runs/20260101-123608/output/test"
    SAVE_PATH = "/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/head/metadata.jsonl"
    
    organize_metadata(TARGET_FOLDER, SAVE_PATH)