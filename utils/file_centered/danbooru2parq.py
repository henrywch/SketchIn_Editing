"""
Used to assembly the head- and character-sketched Danbooru Datasets to Parquets
"""

import json
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = '/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/training_meta/head_captioned.jsonl'       # Replace with your actual input file name
OUTPUT_FILE = '/inspire/hdd/project/video-understanding/public/personal/chwang/live/cpjs/aifl_dif/OminiControl/datasets/danbooru/half/replaced/training_meta/danbooru_insketch_head.parquet'
NUM_WORKERS = cpu_count()       # Uses all available CPU cores

def process_item(args):
    """
    Worker function to process a single line from the jsonl.
    Returns a dictionary row for the dataframe.
    """
    index, line = args
    try:
        data = json.loads(line)
        
        src_image_path = data.get('image_path', '')
        src_cond_path = data.get('conditioning_path', '')
        caption = data.get('caption', '')

        # 1. Extract info
        # Extract filename (e.g., "2511013.png")
        img_filename = os.path.basename(src_image_path)
        cond_filename = os.path.basename(src_cond_path)
        
        # Extract UID (filename without extension, e.g., "2511013")
        uid = os.path.splitext(img_filename)[0]

        # 2. Read Binary Data
        # Ensure the paths in the jsonl are absolute or relative to where you run this script
        with open(src_image_path, 'rb') as f:
            image_binary = f.read()
            
        with open(src_cond_path, 'rb') as f:
            condition_binary = f.read()

        # 3. Construct Data Row
        return {
            "id": index,
            "uid": uid,
            "image": image_binary,
            "condition": condition_binary,
            "image_path": f"datasets/images/{img_filename}",
            "conditioning_path": f"datasets/conditions/{cond_filename}",
            "caption": caption
        }

    except Exception as e:
        # print(f"Error processing line {index}: {e}") # Uncomment to debug specific errors
        return None

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Prepare arguments for multiprocessing (index, line)
    tasks = [(i, line) for i, line in enumerate(lines)]
    total_tasks = len(tasks)
    
    valid_results = []

    print(f"Processing {total_tasks} items using {NUM_WORKERS} processes...")
    
    # Use multiprocessing Pool to read files in parallel
    with Pool(processes=NUM_WORKERS) as pool:
        # imap allows us to iterate over results as they finish for the progress bar
        for result in tqdm(pool.imap(process_item, tasks), total=total_tasks):
            if result is not None:
                valid_results.append(result)

    if not valid_results:
        print("No valid data processed. Exiting.")
        return

    print("Constructing DataFrame...")
    df = pd.DataFrame(valid_results)
    
    # Sort by ID to ensure order is maintained after multiprocessing
    df = df.sort_values('id').reset_index(drop=True)

    print(f"Writing to {OUTPUT_FILE}...")
    
    # Define schema explicitly to ensure binaries are handled correctly
    schema = pa.schema([
        ('id', pa.int64()),
        ('uid', pa.string()),
        ('image', pa.binary()),
        ('condition', pa.binary()),
        ('image_path', pa.string()),
        ('conditioning_path', pa.string()),
        ('caption', pa.string())
    ])

    # Convert to PyArrow Table and write
    table = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(table, OUTPUT_FILE)

    print("Done!")

if __name__ == "__main__":
    main()