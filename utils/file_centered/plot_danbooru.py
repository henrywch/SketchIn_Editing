"""
Plot the Danbooru Scored Results
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np

# ================= CONFIGURATION =================
# Define which metrics are "Higher is Better" vs "Lower is Better"
METRIC_DIRECTIONS = {
    "musiq": "higher",    
    "nima": "higher",     
    "maniqa": "higher",   
    "hyperiqa": "higher", 
    "clipiqa": "higher",  
    "niqe": "lower"       # Lower score = Less distortion
}

def analyze_distributions(file_path):
    print(f"Loading data from {file_path}...")
    
    # Load JSONL line by line
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records.")

    # Setup the plot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    thresholds = {}

    for i, (metric, direction) in enumerate(METRIC_DIRECTIONS.items()):
        ax = axes[i]
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, "Not Found", ha='center')
            continue

        # 1. Filter out Error Codes (-1.0)
        # Your snippet showed maniqa: -1.0. We must exclude these from stats.
        valid_series = df[df[metric] > -0.5][metric]
        
        if len(valid_series) == 0:
            ax.set_title(f"{metric} (No valid data)")
            continue

        # 2. Calculate Statistics
        mean_val = valid_series.mean()
        median_val = valid_series.median()
        
        # 3. Plot Histogram & KDE
        sns.histplot(valid_series, kde=True, ax=ax, color='skyblue', edgecolor='black')
        
        # 4. Draw Lines
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # 5. Determine Threshold (To keep top 50%)
        # If Higher is Better: Threshold is Median. Keep > Median.
        # If Lower is Better: Threshold is Median. Keep < Median.
        thresholds[metric] = median_val
        
        # Add text annotation
        desc = "Higher is Better" if direction == "higher" else "Lower is Better"
        ax.set_title(f"{metric.upper()} ({desc})")
        ax.legend()
        ax.set_xlabel("Score")

    plt.tight_layout()
    plt.savefig("quality_distribution.png")
    print("\nGraph saved to 'quality_distribution.png'")
    
    # --- PRINT REPORT ---
    print("\n" + "="*40)
    print(" RECOMMENDED THRESHOLDS (To keep top 50%)")
    print("="*40)
    
    for metric, val in thresholds.items():
        direction = METRIC_DIRECTIONS[metric]
        operator = ">" if direction == "higher" else "<"
        print(f"{metric.upper():<10} | Median: {val:.4f} | Keep if score {operator} {val:.4f}")
        
    # Check for failure rates (like MANIQA -1.0)
    print("\n" + "="*40)
    print(" FAILURE RATES (-1.0 values)")
    print("="*40)
    for metric in METRIC_DIRECTIONS.keys():
        if metric in df.columns:
            fails = len(df[df[metric] == -1.0])
            pct = (fails / len(df)) * 100
            if pct > 0:
                print(f"{metric}: {fails} images failed ({pct:.2f}%)")

if __name__ == "__main__":
    import json
    # Replace with your actual filename
    analyze_distributions("datasets/danbooru/all_scores.jsonl")