"""
Filtering the VLM BBox SketchyScene Datasets (the Quality of VLM Detected BBoxes)
"""

import os
import json
import asyncio
import base64
import argparse
from typing import List, Dict
from pathlib import Path

# Third-party libraries
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= CONFIGURATION =================
# Update these with your Qwen3-VL deployment details
API_KEY = "" 
BASE_URL = "" 
MODEL_NAME = "Qwen3-VL-235B-A22B-Instruct"

# Concurrency settings
MAX_CONCURRENT_REQUESTS = 20
# =================================================

def encode_image(image_path: str) -> str:
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

async def evaluate_item(
    client: AsyncOpenAI, 
    sem: asyncio.Semaphore, 
    item: Dict, 
    pbar: tqdm_asyncio
) -> Dict:
    """
    Processes a single item: loads images + existing caption, checks correspondence.
    """
    async with sem:
        image_path = item.get("image_path")
        sketch_path = item.get("conditioning_path")
        existing_caption = item.get("caption", "")
        
        # 1. Validation of paths
        if not image_path or not os.path.exists(image_path):
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)
            if not os.path.exists(image_path):
                item["evaluation"] = "Error: Image not found"
                pbar.update(1)
                return item
        
        if not sketch_path or not os.path.exists(sketch_path):
            if not os.path.isabs(sketch_path):
                sketch_path = os.path.join(os.getcwd(), sketch_path)
            if not os.path.exists(sketch_path):
                item["evaluation"] = "Error: Sketch Image not found"
                pbar.update(1)
                return item

        # 2. Encode Images
        base64_image = encode_image(image_path)
        if not base64_image:
            item["evaluation"] = "Error: Image encoding failed"
            pbar.update(1)
            return item
        
        # [FIXED] Use sketch_path here, not image_path
        base64_sketch = encode_image(sketch_path) 
        if not base64_sketch:
            item["evaluation"] = "Error: Sketch encoding failed"
            pbar.update(1)
            return item

        # 3. Construct the Evaluation Prompt
        system_prompt = (
            "You are a data quality assurance expert for computer vision datasets. "
            "Your task is to verify the correspondence between three elements: "
            "1. An Original Image\n"
            "2. A Sketch (Line drawing)\n"
            "3. A Text Caption\n"
            "You must determine if the Sketch is **APPROXIMATELY** a valid structural representation of the Original Image and the Text Caption"
        )

        user_content = [
            {"type": "text", "text": "Original Image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
            {"type": "text", "text": "\nSketch:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_sketch}"},
            },
            {
                "type": "text", 
                "text": f"\nCaption: \"{existing_caption}\"\n\n"
                        "Task: Analyze if the sketch corresponds to the original image and the caption fits. Output 'YES' or 'NO' **ONLY**."
            },
        ]

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=100,
                temperature=0.1, # Low temp for deterministic evaluation
            )
            
            item['eval'] = response.choices[0].message.content.strip().lower()
            
        except Exception as e:
            item['eval'] = f"Error: API call failed - {str(e)}"
            
        pbar.update(1)
        
        return item
        
        

async def main():
    parser = argparse.ArgumentParser(description="Evaluate Sketch-Image-Caption correspondence")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input jsonl containing captions")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save evaluation results")
    args = parser.parse_args()

    # 1. Load Metadata
    print(f"Loading metadata from {args.input_file}...")
    data_items = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_items.append(json.loads(line))
    
    print(f"Loaded {len(data_items)} items.")

    # 2. Initialize OpenAI Client
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 3. Create Tasks
    tasks = []
    pbar = tqdm_asyncio(total=len(data_items), desc="Evaluating Correspondence", position=0)

    for item in data_items:
        task = evaluate_item(client, sem, item, pbar)
        tasks.append(task)

    # 4. Run Async Loop
    results = await asyncio.gather(*tasks)
    results = [result for result in results if result]
    pbar.close()

    # 5. Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving results to {args.output_dir}...")
    
    valid_count = len(results)
    invalid_cases = []
    
    valid_output_path = os.path.join(args.output_dir, "evaluated.jsonl")
    invalid_output_path = os.path.join(args.output_dir, "invalid.jsonl")
    
    with open(valid_output_path, 'w', encoding='utf-8') as f:
        for item in results:
            if item['eval'] == 'yes':
                del item['eval']
                f.write(json.dumps(item) + "\n")
            else: invalid_cases.append(item)
    with open(invalid_output_path, 'w', encoding='utf-8') as f:
        json.dump(invalid_cases, f, ensure_ascii=False, indent=4)

    print(f"Done! Processed {len(results)} items. {valid_count} successful API calls.")
    print(f"Results saved to {valid_output_path}")

if __name__ == "__main__":
    asyncio.run(main())