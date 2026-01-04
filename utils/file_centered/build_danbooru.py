"""
Used to download Danbooru Datasets
"""

import os
import shutil
import kagglehub

# --- CONFIGURATION ---
os.environ["KAGGLE_USERNAME"] = "luohe_54"
os.environ["KAGGLE_KEY"] = "KGAT_4763052ab1504963d24079176f3b7dd7"

# --- DOWNLOADING ---
target_path = "datasets/danbooru"

print("Authenticating and downloading...")
try:
    # The download will now use the credentials set above
    cache_path = kagglehub.dataset_download("wuhecong/danbooru-sketch-pair-128x")
    print(f"Initially downloaded to cache: {cache_path}")

    # Ensure target directory is clean
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    # Move files
    shutil.move(cache_path, target_path)
    print(f"Final dataset location: {os.path.abspath(target_path)}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Check if your KAGGLE_USERNAME and KAGGLE_KEY are correct.")