"""
Provide content-rich caption to the SketchyScene Datasets
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
BASE_URL = "" # e.g., vLLM or SGLang endpoint
MODEL_NAME = "Qwen3-VL-235B-A22B-Instruct" # Change to your deployed model name (e.g. Qwen3-Vl-235B...)

# Concurrency settings
MAX_CONCURRENT_REQUESTS = 20  # Adjust based on your API rate limits
# =================================================

def encode_image(image_path: str) -> str:
    """Encodes a local image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

async def generate_caption_for_item(
    client: AsyncOpenAI, 
    sem: asyncio.Semaphore, 
    item: Dict, 
    pbar: tqdm_asyncio
) -> Dict:
    """
    Processes a single dataset item: loads image, calls API, updates caption.
    """
    async with sem:
        image_path = item.get("image_path")
        sketch_path = item.get("conditioning_path")
        
        # 1. Validation
        if not image_path or not os.path.exists(image_path):
            # If absolute path fails, try relative to script execution
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)
            
            if not os.path.exists(image_path):
                item["caption"] = "Error: Image not found"
                pbar.update(1)
                return item
        
        if not sketch_path or not os.path.exists(sketch_path):
            # If absolute path fails, try relative to script execution
            if not os.path.isabs(sketch_path):
                sketch_path = os.path.join(os.getcwd(), sketch_path)
            
            if not os.path.exists(sketch_path):
                item["caption"] = "Error: Sketch Image not found"
                pbar.update(1)
                return item

        # 2. Encode Image
        base64_image = encode_image(image_path)
        if not base64_image:
            item["caption"] = "Error: Image encoding failed"
            pbar.update(1)
            return item
        
        base64_sketch = encode_image(sketch_path)
        if not base64_sketch:
            item["caption"] = "Error: Sketch encoding failed"
            pbar.update(1)
            return item

        # 3. Construct the Prompt
        # We define the system style using the FSCOCO examples provided.
        # We only send the RGB Image (reference_image). Sending the sketch is unnecessary 
        # for generating a semantic caption of the scene content and might confuse the model
        # into describing the sketch style.
        
        system_prompt = (
            "You are an intelligent image captioning expert. "
            "Your task is to provide a concise, objective description of the scene content for sketch drawing task, which means to generate a picture based on a sketch provided."
            "As the sketch mainly draws the main object in the image, please focus on captioning the main objects and their spatial relationships. "
            "Do NOT mention that this is a 'cartoon', 'drawing', or 'image'. "
            "Describe the scene as if it were real."
            "\n\nStyle Examples:"
            "\n- A fire hydrant is located in the middle of grassy field"
            "\n- Street sign showed in the picture"
        )

        user_prompt = (
            "Original Image: <image>",
            "Sketch: <image>",
            "Please output the concise caption **ONLY**"
            )

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            # Interleaved text and images
                            {"type": "text", "text": "Original Image:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": "\nSketch:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_sketch}"
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ],
                max_tokens=100,
                temperature=0.2, # Low temperature for consistent, descriptive captions
            )
            
            generated_caption = response.choices[0].message.content.strip()
            item["caption"] = generated_caption
            
        except Exception as e:
            # print(f"API Error for {image_path}: {e}")
            item["caption"] = "Error: API call failed"
        
        pbar.update(1)
        return item

async def main():
    parser = argparse.ArgumentParser(description="Generate captions for SketchyScene metadata")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input metadata.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output metadata.jsonl")
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
    
    # Semaphore to limit concurrency (prevents rate limiting)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 3. Create Tasks
    tasks = []
    # Create the progress bar
    pbar = tqdm_asyncio(total=len(data_items), desc="Captioning Images", position=0)

    for item in data_items:
        task = generate_caption_for_item(client, sem, item, pbar)
        tasks.append(task)

    # 4. Run Async Loop
    results = await asyncio.gather(*tasks)
    
    pbar.close()

    # 5. Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving results to {args.output_dir}...")
    invalid = []
    valid_count = 0
    with open(os.path.join(args.output_dir, "caped.jsonl"), 'w', encoding='utf-8') as f:
        for item in results:
            # Filter out errors if you want, or keep them for debugging
            if item['caption'] and not item["caption"].startswith("Error:"):
                valid_count += 1
                f.write(json.dumps(item) + "\n")
            else: 
                invalid.append(item)
                
    with open(os.path.join(args.output_dir, "invalid.json"), 'w', encoding='utf-8') as f:
        json.dump(invalid, f, ensure_ascii=False, indent=4)

    print(f"Done! Processed {len(results)} items. {valid_count} valid captions.")

if __name__ == "__main__":
    asyncio.run(main())