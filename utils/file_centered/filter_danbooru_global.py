"""
Filter the Danbooru Datasets Globally with the average scores of all the PYIQA scores
"""

import json
import pandas as pd
import argparse
import os

# ================= CONFIGURATION =================
# We skip 'maniqa' as requested due to failures
METRICS_TO_USE = ["musiq", "nima", "hyperiqa", "clipiqa"] 
INVERT_METRIC = "niqe" # Lower is better, so we invert it

def filter_dataset(input_file, output_file, top_percent=0.5):
    print(f"Reading {input_file}...")
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Basic validation: ensure all required keys exist and are not error codes (-1)
                valid = True
                for m in METRICS_TO_USE + [INVERT_METRIC]:
                    if m not in item or item[m] == -1.0:
                        valid = False
                        break
                if valid:
                    data.append(item)
            except:
                continue
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} valid entries.")

    # --- 1. NORMALIZATION ---
    # Create a new dataframe for normalized scores
    norm_df = pd.DataFrame()

    # Normalize "Higher is Better" metrics (Min-Max Normalization)
    for metric in METRICS_TO_USE:
        min_val = df[metric].min()
        max_val = df[metric].max()
        # Avoid division by zero
        if max_val - min_val == 0:
            norm_df[f"norm_{metric}"] = 0.0
        else:
            norm_df[f"norm_{metric}"] = (df[metric] - min_val) / (max_val - min_val)

    # Normalize "Lower is Better" metric (NIQE)
    # Logic: 1 - (Score / Max) is a simple inversion, but Min-Max inversion is often more stable:
    # Inverted Min-Max: (Max - Score) / (Max - Min)
    niqe_max = df[INVERT_METRIC].max()
    niqe_min = df[INVERT_METRIC].min()
    
    if niqe_max - niqe_min == 0:
        norm_df[f"norm_{INVERT_METRIC}"] = 0.0
    else:
        # We use standard Min-Max inversion so it scales 0.0 to 1.0 like the others
        norm_df[f"norm_{INVERT_METRIC}"] = (niqe_max - df[INVERT_METRIC]) / (niqe_max - niqe_min)

    # --- 2. COMPOSITE SCORE CALCULATION ---
    # Average of MUSIQ, NIMA, HYPER, CLIP, and Inverted NIQE
    # Total metrics = 4 (standard) + 1 (inverted) = 5
    print("Calculating Composite Scores...")
    
    # Sum all columns in norm_df and divide by count
    df["final_score"] = norm_df.sum(axis=1) / len(norm_df.columns)

    # --- 3. FILTERING ---
    # Sort by Final Score descending (Best first)
    df_sorted = df.sort_values(by="final_score", ascending=False)

    # Calculate cutoff index
    cutoff_index = int(len(df_sorted) * top_percent)
    
    # Keep top 50%
    df_kept = df_sorted.iloc[:cutoff_index]
    df_discarded = df_sorted.iloc[cutoff_index:]

    print("-" * 30)
    print(f"Total Images: {len(df)}")
    print(f"Kept Images:  {len(df_kept)} (Top {top_percent*100}%)")
    print(f"Score Threshold: > {df_kept['final_score'].min():.4f}")
    print("-" * 30)

    # --- 4. SAVING ---
    print(f"Saving to {output_file}...")
    
    # We convert back to a list of dicts, keeping original keys + final_score
    records_to_save = df_kept.to_dict(orient="records")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in records_to_save:
            f.write(json.dumps(row) + "\n")

    # Optional: Save discarded for inspection
    discard_path = output_file.replace(".jsonl", "_discarded.jsonl")
    records_discarded = df_discarded.to_dict(orient="records")
    with open(discard_path, 'w', encoding='utf-8') as f:
        for row in records_discarded:
            f.write(json.dumps(row) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to all_scores.jsonl")
    parser.add_argument("--output", type=str, default="filtered_dataset.jsonl", help="Path to save filtered results")
    parser.add_argument("--ratio", type=float, default=0.5, help="Ratio to keep (0.5 = 50%)")
    args = parser.parse_args()

    if os.path.exists(args.input):
        filter_dataset(args.input, args.output, args.ratio)
    else:
        print("Input file not found.")