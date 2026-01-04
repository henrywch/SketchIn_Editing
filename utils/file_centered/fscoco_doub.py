"""
Apply Non-Captioned Entries into the FSCOCO Metadata
"""

import json
import random

INPUT_FILE = "/inspire/hdd/project/video-understanding/public/personal/chwang/datasets/fscoco/fscoco/metadata.jsonl"
OUTPUT_FILE = "/inspire/hdd/project/video-understanding/public/personal/chwang/datasets/fscoco/fscoco/metadata_d.jsonl"

all_entries = []

print(f"Reading from {INPUT_FILE}...")

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        line = line.strip()
        if not line: continue

        original_dict = json.loads(line)

        no_caption_dict = original_dict.copy()
        no_caption_dict['caption'] =  ""

        all_entries.append(original_dict)
        all_entries.append(no_caption_dict)

print(f"Shuffling {len(all_entries)} entries...")
random.shuffle(all_entries)

print(f"Writing to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    for entry in all_entries:
        f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("Done!")