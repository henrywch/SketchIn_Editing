"""
Call SAM3 to detect Heads and Characters in Danbooru Datasets
"""

import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import warnings
import logging
from PIL import Image

from ultralytics.models.sam import SAM3SemanticPredictor

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
TEXT_PROMPTS = ["anime character", "person", "head", "face"]
CLASS_MAPPING = {
    0: "character",
    1: "character",
    2: "head",
    3: "head"
}

CONF_THRESHOLD = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM3_MODEL_PATH = "/inspire/hdd/project/video-understanding/public/share/models/SAM3/sam3.pt" 

# -----------------------------------------------------------------------------
# INITIALIZATION & UTILITIES
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def load_image(path):
    """
    Loads an image using PIL to avoid libpng iCCP warnings from OpenCV.
    Returns a BGR numpy array (standard OpenCV format).
    """
    try:
        # PIL handles bad iCCP profiles silently
        pil_img = Image.open(path).convert("RGB") 
        img_rgb = np.array(pil_img)
        # Convert RGB (PIL) to BGR (OpenCV)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def initialize_predictor():
    """
    Initializes the Ultralytics SAM3SemanticPredictor.
    """
    print(f"Initializing SAM 3 Semantic Predictor on {DEVICE}...")

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=SAM3_MODEL_PATH,
        half=True,
        save=False
    )
    
    # Initialize the predictor class directly
    return SAM3SemanticPredictor(overrides=overrides)

def draw_bboxes(img, bbox_dict, thickness=1, color=None):
    """Draws bounding boxes on an image and returns the numpy array."""
    img_copy = img.copy()
    
    DEFAULT_COLORS = {
        "character": (255, 0, 0),  # Blue
        "head":      (0, 0, 255)   # Red
    }

    for category, boxes in bbox_dict.items():
        draw_color = color if color is not None else DEFAULT_COLORS.get(category, (0, 255, 0))
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), draw_color, thickness)
            cv2.putText(img_copy, category, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, draw_color, 2)
    
    return img_copy

# -----------------------------------------------------------------------------
# MAIN PROCESSING LOGIC
# -----------------------------------------------------------------------------
def process_dataset(input_jsonl, output_jsonl):
    # 1. Setup
    predictor = initialize_predictor()
    base_dir = os.path.dirname(output_jsonl)
    
    dirs = {
        "img": os.path.join(base_dir, "images_bboxed"),
        "sketch": os.path.join(base_dir, "sketches_bboxed")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 2. Load Data
    print(f"Reading {input_jsonl}...")
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # 3. Processing Loop
    valid_count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for entry in tqdm(data, desc="Processing Pairs"):
            item_id = entry.get("id")
            
            # Path Handling (Absolute conversion)
            paths = {
                "img": os.path.abspath(entry.get("image")) if not os.path.isabs(entry.get("image")) else entry.get("image"),
                "sketch": os.path.abspath(entry.get("sketch")) if not os.path.isabs(entry.get("sketch")) else entry.get("sketch")
            }

            # File Validation
            if not all(os.path.exists(p) for p in paths.values()):
                continue

            try:
                entry_results = {
                    "img_bbox_coord": {"character": [], "head": []},
                    "sketch_bbox_coord": {"character": [], "head": []},
                    "saved_files": {}
                }
                
                has_any_detection = False 
                
                for key_type in ["img", "sketch"]:
                    current_path = paths[key_type]
                    
                    # 1. Set image for THIS specific file (image or sketch)
                    image_array = load_image(current_path)
                    if image_array is None:
                        continue
                    
                    predictor.set_image(image_array)
                    
                    # 2. Predict
                    results = predictor(text=TEXT_PROMPTS)
                    
                    # Container for this specific file's boxes
                    grouped_bboxes = {
                        "character": [],
                        "head": []
                    }

                    if results and results[0].boxes is not None:
                        result = results[0]
                        boxes_data = result.boxes.xyxy.cpu().numpy().tolist()
                        scores_data = result.boxes.conf.cpu().numpy().tolist()
                        class_indices = result.boxes.cls.cpu().numpy().tolist()

                        # Filter and Group
                        for box, score, cls_idx in zip(boxes_data, scores_data, class_indices):
                            if score > CONF_THRESHOLD:
                                category_key = CLASS_MAPPING.get(int(cls_idx))
                                if category_key:
                                    grouped_bboxes[category_key].append(box)
                    
                    # Store coordinates in the requested key format
                    entry_results[f"{key_type}_bbox_coord"] = grouped_bboxes
                    
                    # Track if we found anything at all for this entry
                    if grouped_bboxes["character"] and grouped_bboxes["head"]:
                        has_any_detection = True

                    # 3. Visualization & Saving (Specific to this file)
                    # Define colors: Green for img, Red for sketch
                    colors = {"img": (0, 255, 0), "sketch": (0, 0, 255)}
                    
                    viz = draw_bboxes(image_array, grouped_bboxes, color=colors[key_type])
                    
                    save_name = f"{item_id}_{key_type}_bbox.jpg"
                    save_abs = os.path.join(dirs[key_type], save_name)
                    cv2.imwrite(save_abs, viz)
                    entry_results["saved_files"][f"{key_type}_bboxed"] = save_abs

                if not has_any_detection:
                     continue

                out_entry = {
                    "id": item_id,
                    "image": paths["img"],
                    "sketch": paths["sketch"],
                    "img_bbox_coord": entry_results["img_bbox_coord"],
                    "sketch_bbox_coord": entry_results["sketch_bbox_coord"],
                    **entry_results["saved_files"]
                }
                f_out.write(json.dumps(out_entry) + "\n")
                valid_count += 1

            except Exception as e:
                print(f"\n[Error] {item_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"Done! Processed {valid_count} entries.")
    print(f"Saved to {output_jsonl}")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE = "datasets/danbooru/half/final_paired.jsonl"
    OUTPUT_FILE = "datasets/danbooru/half/scored.jsonl"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    process_dataset(INPUT_FILE, OUTPUT_FILE)