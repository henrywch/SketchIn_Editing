"""
Used to generate metadata for VLM BBoxed SketchScene Datasets
"""

import json
import os
from tqdm import tqdm

def merge_bboxes(file1_path, file2_path, output_path):
    # Dictionary to store mapping: {extracted_id: bboxes}
    # We use a dictionary for O(1) lookup speed instead of nested loops
    bbox_lookup = {}

    print(f"Reading {file1_path} and indexing bboxes...")
    with open(file1_path, 'r', encoding='utf-8') as f1:
        # Count lines for tqdm total
        lines_f1 = f1.readlines()
        for line in tqdm(lines_f1, desc="Indexing File 1"):
            data = json.loads(line)
            img_path = data.get("image_path", "")
            bboxes = data.get("bboxes", [])
            
            # To make the "uuid in image_path" check efficient, 
            # we extract the filename or the ID part.
            # Example: "datasets/.../477/1226_image_bbox.jpg" -> "1226"
            filename = os.path.basename(img_path)
            bbox_lookup[filename] = bboxes

    results = []
    print(f"Processing {file2_path} and matching UUIDs...")
    with open(file2_path, 'r', encoding='utf-8') as f2:
        lines_f2 = f2.readlines()
        
        for line in tqdm(lines_f2, desc="Updating File 2"):
            item = json.loads(line)
            uuid = str(item.get("uuid", ""))
            
            # The user logic: "whether the uuid is in the image_path of an entry"
            # We check our lookup table keys (filenames from File 1)
            found_bbox = None
            for filename, bboxes in bbox_lookup.items():
                if uuid in filename:
                    found_bbox = bboxes
                    break
            
            # If a match was found, add the bboxes key
            if found_bbox is not None:
                item["bboxes"] = found_bbox
            else:
                # Optional: Handle cases where uuid isn't found
                item["bboxes"] = [] 

            results.append(item)

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for entry in tqdm(results, desc="Writing JSONL"):
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("Done!")

if __name__ == "__main__":
    # Update these filenames to your actual file paths
    FILE_1 = "datasets/bboxed_sketchyscene/metadata.jsonl" # The one with bboxes
    FILE_2 = "datasets/SS_bbox_filtered/metadata.jsonl" # The one with uuids
    OUTPUT_FILE = "datasets/SS_bbox_filtered/metadata_bboxed.jsonl"

    merge_bboxes(FILE_1, FILE_2, OUTPUT_FILE)