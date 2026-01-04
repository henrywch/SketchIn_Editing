"""
Filter the bbox cast by the VLMs
"""

import os
import shutil
import base64
import asyncio
import argparse
import re
from pathlib import Path
from typing import Tuple, Optional

# Third-party libs
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= CONFIGURATION =================
# Update these with your specific server details
API_KEY = "" 
BASE_URL = "" 
MODEL_NAME = "Qwen3-VL-235B-A22B-Instruct"

# How many requests to send in parallel
MAX_CONCURRENT_REQUESTS = 20 
# =================================================

def encode_image(image_path: str) -> Optional[str]:
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return None

def find_image_pairs(root_dir: str) -> list[Tuple[str, str, str]]:
    """
    Scans the directory structure: root/subdir/ID_image_bbox.jpg & ID_sketch_bbox.jpg
    Returns a list of tuples: (id_stem, path_to_image, path_to_sketch)
    """
    pairs = []
    root = Path(root_dir)
    
    # We look for files ending in _image_bbox.jpg
    # Assuming structure: root/subdir/000000494112_image_bbox.jpg
    for img_path in root.rglob("*_image_bbox.jpg"):
        # Construct expected sketch path
        # Name: 000000494112_image_bbox.jpg -> 000000494112_sketch_bbox.jpg
        # print(img_path)
        file_id = img_path.name.split('_')[0]
        sketch_name = f"L0_sample{file_id}_sketch_bbox.png"
        sketch_path = img_path.parent / sketch_name
        
        if sketch_path.exists():
            # ID is the filename prefix (e.g., 000000494112)
            pairs.append((file_id, str(img_path), str(sketch_path)))
            
    return pairs

async def process_pair(
    client: AsyncOpenAI, 
    sem: asyncio.Semaphore, 
    pair: Tuple[str, str, str], 
    output_dir: Path,
    pbar: tqdm_asyncio
):
    """
    Sends request to VLM and copies files if answer is YES.
    """
    file_id, img_path, sketch_path = pair
    
    async with sem:
        # 1. Encode Images
        b64_img = encode_image(img_path)
        b64_sketch = encode_image(sketch_path)
        
        if not b64_img or not b64_sketch:
            pbar.update(1)
            return

        # 2. Construct Prompt
        system_prompt = (
            "You are a strict data quality assistant. You will be shown two images: "
            "a real photo and a corresponding sketch. Both have bounding boxes drawn on them."
        )
        
        user_text = (
            "Please analyze the bounding boxes in these two images based on two criteria:\n"
            "1. **Correspondence**: Do the bounding boxes in the photo and the sketch refer to the exact same object instance?\n"
            "2. **Quality**: Is the bounding box reasonably tight (almost contacting the object edges) AND does it encompass the *main* salient object of the image/sketch?\n\n"
            "Answer with 'YES' only if ALL conditions are met. Otherwise answer 'NO'. Do not explain."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Image 1 (Photo):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                {"type": "text", "text": "Image 2 (Sketch):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_sketch}"}},
                {"type": "text", "text": user_text}
            ]}
        ]

        try:
            # 3. Call API
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=10,
                temperature=0.01, # Deterministic
            )
            
            answer = response.choices[0].message.content.strip().upper()
            
            # 4. Filter and Copy
            # Check if answer contains "YES" (handling potential punctuation like "Yes.")
            if "YES" in answer and "NO" not in answer:
                # Maintain subdirectory structure in output? Or flat? 
                # Let's keep it flat or grouped by ID for simplicity, or mirror parent name.
                # Here we mirror the immediate parent folder name to avoid collisions.
                parent_name = Path(img_path).parent.name
                target_folder = output_dir / parent_name
                target_folder.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(img_path, target_folder / Path(img_path).name)
                shutil.copy2(sketch_path, target_folder / Path(sketch_path).name)

        except Exception as e:
            # print(f"Error processing {file_id}: {e}")
            pass
        finally:
            pbar.update(1)

async def main():
    parser = argparse.ArgumentParser(description="Filter bbox pairs using VLM")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing source images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save accepted pairs")
    args = parser.parse_args()

    # 1. Scan Files
    print(f"Scanning {args.input_dir}...")
    pairs = find_image_pairs(args.input_dir)
    print(f"Found {len(pairs)} pairs.")
    
    if not pairs:
        return

    # 2. Setup
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks = []
    pbar = tqdm_asyncio(total=len(pairs), desc="Filtering Pairs")

    # 3. Queue Tasks
    for pair in pairs:
        task = process_pair(client, sem, pair, out_path, pbar)
        tasks.append(task)

    # 4. Run
    await asyncio.gather(*tasks)
    pbar.close()
    print(f"Done. Qualified images saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())