"""
Filter the Danbooru Datasets Stagely with each PYIQA scores
"""

import json
import pandas as pd
import argparse
import os

# ================= CONFIGURATION =================
# Define metrics and their "Good" direction
# "high" = Higher is Better (Keep > Median)
# "low"  = Lower is Better  (Keep < Median)
METRICS_CONFIG = {
    "musiq": "high",
    "nima": "high",
    "hyperiqa": "high",
    "clipiqa": "high",
    "niqe": "low"
}

def filter_sequential(input_file, output_file):
    print(f"Reading {input_file}...")
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Basic validation: ensure keys exist and are not -1.0
                valid = True
                for m in METRICS_CONFIG.keys():
                    if m not in item or item[m] == -1.0:
                        valid = False
                        break
                if valid:
                    data.append(item)
            except:
                continue
    
    df = pd.DataFrame(data)
    initial_count = len(df)
    print(f"Loaded {initial_count} valid entries.")
    print("-" * 50)

    # We make a copy to filter iteratively
    df_filtered = df.copy()

    # --- SEQUENTIAL FILTERING LOOP ---
    for metric, direction in METRICS_CONFIG.items():
        # 1. Calculate Median on the ORIGINAL full dataset
        # (It is important to use the global median, not the median of the remaining subset,
        # otherwise you keep slicing the dataset in half indefinitely)
        global_median = df[metric].median()
        
        prev_count = len(df_filtered)
        
        # 2. Apply Filter
        if direction == "high":
            # Keep items GREATER than or equal to median
            df_filtered = df_filtered[df_filtered[metric] >= global_median]
            symbol = ">="
        else:
            # Keep items LESS than or equal to median (for NIQE)
            df_filtered = df_filtered[df_filtered[metric] <= global_median]
            symbol = "<="
            
        curr_count = len(df_filtered)
        dropped = prev_count - curr_count
        
        print(f"Filtering by {metric.upper():<10} | Threshold: {symbol} {global_median:.4f}")
        print(f"   -> Dropped: {dropped} images")
        print(f"   -> Remaining: {curr_count} ({(curr_count/initial_count)*100:.2f}%)")
        print("-" * 50)

    # --- SAVING ---
    print(f"Final Selection: {len(df_filtered)} images out of {initial_count}")
    print(f"Total Retention Rate: {(len(df_filtered)/initial_count)*100:.2f}%")
    
    print(f"Saving to {output_file}...")
    records = df_filtered.to_dict(orient="records")
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
            
    # Save excluded for inspection
    discarded_path = output_file.replace(".jsonl", "_discarded.jsonl")
    # Get IDs of kept images
    kept_indices = df_filtered.index
    # Filter original DF for indices NOT in kept
    df_discarded = df[~df.index.isin(kept_indices)]
    
    with open(discarded_path, 'w', encoding='utf-8') as f:
        for row in df_discarded.to_dict(orient="records"):
            f.write(json.dumps(row) + "\n")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to all_scores.jsonl")
    parser.add_argument("--output", type=str, default="strict_filtered.jsonl", help="Path to save result")
    args = parser.parse_args()

    if os.path.exists(args.input):
        filter_sequential(args.input, args.output)
    else:
        print("Input file not found.")