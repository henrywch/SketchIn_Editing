"""
Generate Metadata for Danbooru Datasets
"""

import json
import os
import random
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_JSONL = "/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/combined_meta.jsonl"
OUTPUT_DIR = "/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/training_meta"

# Fixed Prompt
CAPTION_TEXT = "Please fill in the image according to the sketch in the image"

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def write_jsonl(data, path, desc):
    """Writes a list of dicts to a JSONL file with a progress bar."""
    print(f"Saving {desc} to {os.path.basename(path)}...")
    with open(path, 'w', encoding='utf-8') as f:
        for entry in tqdm(data, desc=desc):
            f.write(json.dumps(entry) + "\n")

# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------
def generate_metadata(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Input Data
    print(f"Reading {input_path}...")
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Containers for Head and Character datasets
    head_captioned = []
    head_empty = []
    
    char_captioned = []
    char_empty = []

    # 2. Process Entries
    for entry in tqdm(entries, desc="Processing Entries"):
        # Common original image is the target "image_path"
        # The combined image is the "conditioning_path" (the input to the model)
        # Note: Usually in control/inpainting tasks:
        #   target = original image (ground truth)
        #   source = conditioning image (sketch+image composite)
        
        orig_image = entry.get("image")
        
        # --- HEAD DATA ---
        head_combined = entry.get("head_combined")
        if head_combined and os.path.exists(head_combined):
            # Captioned Entry
            head_captioned.append({
                "image_path": orig_image,
                "conditioning_path": head_combined,
                "caption": CAPTION_TEXT
            })
            # Empty Caption Entry
            head_empty.append({
                "image_path": orig_image,
                "conditioning_path": head_combined,
                "caption": ""
            })

        # --- CHARACTER DATA ---
        char_combined = entry.get("character_combined")
        if char_combined and os.path.exists(char_combined):
            # Captioned Entry
            char_captioned.append({
                "image_path": orig_image,
                "conditioning_path": char_combined,
                "caption": CAPTION_TEXT
            })
            # Empty Caption Entry
            char_empty.append({
                "image_path": orig_image,
                "conditioning_path": char_combined,
                "caption": ""
            })

    # 3. Generate Mixed & Shuffled Datasets
    # Head
    head_mixed = head_captioned + head_empty
    random.shuffle(head_mixed)
    
    # Character
    char_mixed = char_captioned + char_empty
    random.shuffle(char_mixed)

    # 4. Save All Files
    # --- HEADS ---
    write_jsonl(head_captioned, os.path.join(output_dir, "head_captioned.jsonl"), "Head (Captioned)")
    write_jsonl(head_empty,     os.path.join(output_dir, "head_empty.jsonl"),     "Head (Empty)")
    write_jsonl(head_mixed,     os.path.join(output_dir, "head_mixed.jsonl"),     "Head (Mixed)")

    # --- CHARACTERS ---
    write_jsonl(char_captioned, os.path.join(output_dir, "char_captioned.jsonl"), "Char (Captioned)")
    write_jsonl(char_empty,     os.path.join(output_dir, "char_empty.jsonl"),     "Char (Empty)")
    write_jsonl(char_mixed,     os.path.join(output_dir, "char_mixed.jsonl"),     "Char (Mixed)")

    print("\nGeneration Complete!")
    print(f"Total Heads: {len(head_captioned)} unique pairs")
    print(f"Total Characters: {len(char_captioned)} unique pairs")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_metadata(INPUT_JSONL, OUTPUT_DIR)