"""
Simple codes to call VLMs to BBox images
"""

import os
import json
import asyncio
import base64
import re
from typing import List, Dict, Union, Any

from openai import AsyncOpenAI
from PIL import Image, ImageDraw
from tqdm.asyncio import tqdm

# ================= CONFIGURATION =================
# Replace with your actual API endpoint and Key
API_KEY = "" 
BASE_URL = "" 
MODEL_NAME = "Qwen3-VL-235B-A22B-Instruct"

# Concurrency limit
MAX_CONCURRENT_REQUESTS = 10 

# Paths
# INPUT_JSONL = "/inspire/hdd/project/video-understanding/public/personal/chwang/datasets/fscoco/fscoco/metadata.jsonl"
INPUT_JSONL = "datasets/sketchyscene_recaped/recaped.jsonl"
OUTPUT_DIR = "datasets/bboxed_sketchyscene"
METADATA_FILENAME = "metadata.jsonl"
# =================================================

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path: str) -> str:
    """Encodes a local image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_coordinates(text: str) -> List[List[float]]:
    """
    Parses normalized coordinates from text using regex.
    Expects format like [0.1, 0.2, 0.3, 0.4].
    """
    matches = re.findall(r"\[([0-9\.\,\s]+)\]", text)
    results = []
    for match in matches:
        try:
            coords = [float(x.strip()) for x in match.split(',')]
            if len(coords) == 4:
                results.append(coords)
        except ValueError:
            continue
    return results

def draw_bbox_on_image(image_path: str, bbox: List[float], suffix: str, save_dir: str) -> Union[str, None]:
    """
    Draws the bbox on the image, saves it, and returns the new file path.
    """
    try:
        if not bbox or len(bbox) != 4:
            return None

        with Image.open(image_path).convert("RGB") as img:
            width, height = img.size
            ymin, xmin, ymax, xmax = bbox

            # Denormalize
            left = xmin * width
            top = ymin * height
            right = xmax * width
            bottom = ymax * height

            draw = ImageDraw.Draw(img)
            # Draw a thick red bounding box
            draw.rectangle([left, top, right, bottom], outline="red", width=4)

            # Construct output filename
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            
            # Example filename: 000000566027_image_bbox.jpg
            new_filename = f"{name}_{suffix}{ext}"
            save_path = os.path.join(save_dir, new_filename)
            
            img.save(save_path)
            return save_path
    except Exception as e:
        print(f"Error drawing on {image_path}: {e}")
        return None

async def process_row(sem: asyncio.Semaphore, row: dict, line_id: int) -> Dict[str, Any]:
    """
    Process a single JSONL row. 
    Returns a dict with 'status' ('success' or 'error') and relevant data.
    """
    async with sem:
        img_path = row.get("image_path")
        sketch_path = row.get("conditioning_path")
        caption = row.get("recap")

        # Basic validation
        if not img_path or not sketch_path:
             return {"status": "error", "msg": f"Line {line_id}: Missing paths in input."}

        if not os.path.exists(img_path) or not os.path.exists(sketch_path):
            return {"status": "error", "msg": f"Line {line_id}: Files not found on disk."}

        try:
            # Encode images
            img_b64 = encode_image(img_path)
            sketch_b64 = encode_image(sketch_path)

            # Construct Prompt
            sys_prompt = (
                "You are a helpful vision assistant skilled in object detection."
                f"Identify the object described as **{caption}** in **both the provided Image and the Sketch**.\n"
                "Return **the Bounding Box (BBox) of the object** for the Image first, followed by **the BBox of the object** for the Sketch.\n"
                "Format required: [ymin, xmin, ymax, xmax]\n"
                "Ensure coordinates are normalized (0.0 to 1.0).\n\n"
                "Output ONLY:\n"
                "Image: [y, x, y, x]\n"
                "Sketch: [y, x, y, x]"
            )
            user_prompt = (
                "Image: <image>"
                "Sketch: <image>"
            )

            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{sketch_b64}"}},
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                max_tokens=200,
                temperature=0.0
            )

            content = response.choices[0].message.content
            bboxes = parse_coordinates(content)

            if len(bboxes) >= 2:
                img_bbox = bboxes[0]
                sketch_bbox = bboxes[1]

                # Create subfolder based on line_id to avoid name collisions
                sub_dir = os.path.join(OUTPUT_DIR, str(line_id))
                os.makedirs(sub_dir, exist_ok=True)
                
                # Draw bboxes and get new paths
                new_img_path = draw_bbox_on_image(img_path, img_bbox, "image_bbox", sub_dir)
                new_sketch_path = draw_bbox_on_image(sketch_path, sketch_bbox, "sketch_bbox", sub_dir)
                
                if new_img_path and new_sketch_path:
                    # Return success data
                    return {
                        "status": "success",
                        "data": {
                            "image_path": new_img_path,
                            "conditioning_path": new_sketch_path,
                            "caption": caption,
                            "bboxes": bboxes
                        }
                    }
                else:
                     return {"status": "error", "msg": f"Line {line_id}: Failed to save images."}
            else:
                return {"status": "error", "msg": f"Line {line_id}: Failed to parse 2 bboxes. Output: {content}"}

        except Exception as e:
            return {"status": "error", "msg": f"Line {line_id}: Exception: {str(e)}"}

async def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Read input file
    print(f"Reading {INPUT_JSONL}...")
    try:
        with open(INPUT_JSONL, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {INPUT_JSONL} not found.")
        return

    # Prepare tasks
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    print(f"Processing {len(lines)} entries...")
    
    for i, line in enumerate(lines):
        if not line.strip(): continue
        row = json.loads(line)
        tasks.append(process_row(semaphore, row, i))

    # Run and collect results
    success_entries = []
    error_messages = []

    # Using tqdm for progress bar
    for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
        result = await f
        if result["status"] == "success":
            success_entries.append(result["data"])
        else:
            error_messages.append(result["msg"])

    # Write metadata.jsonl
    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILENAME)
    print(f"\nWriting metadata to {metadata_path}...")
    
    with open(metadata_path, 'w') as f:
        for entry in success_entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    print(f"\n--- Processing Complete ---")
    print(f"Successful: {len(success_entries)}")
    print(f"Failed:     {len(error_messages)}")
    
    if error_messages:
        print("\n--- Error Log ---")
        for err in error_messages[:10]: # Print first 10 errors
            print(err)
        if len(error_messages) > 10:
            print(f"... and {len(error_messages) - 10} more errors.")

if __name__ == "__main__":
    asyncio.run(main())