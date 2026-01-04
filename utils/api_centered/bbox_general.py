"""
An Attemp to generalize SAM3/SAM Object Segmentation to FSCOCO & SketchyScene Datasets (Unfinished)
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

# Assuming this import exists in your environment
from ultralytics.models.sam import SAM3SemanticPredictor

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# REMOVED: TEXT_PROMPTS and CLASS_MAPPING constraints
# We will now accept all detections.

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
        pil_img = Image.open(path).convert("RGB") 
        img_rgb = np.array(pil_img)
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
    
    return SAM3SemanticPredictor(overrides=overrides)

def draw_bboxes(img, bbox_dict, thickness=2, color=None):
    """Draws bounding boxes on an image and returns the numpy array."""
    img_copy = img.copy()
    
    # Generic default color (Green)
    DEFAULT_COLOR = (0, 255, 0)

    for category, boxes in bbox_dict.items():
        # Use provided color or default
        draw_color = color if color is not None else DEFAULT_COLOR
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), draw_color, thickness)
            
            # Simple label since we are detecting everything
            label = f"{category}"
            cv2.putText(img_copy, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
    
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
            
            paths = {
                "img": os.path.abspath(entry.get("image_path")) if not os.path.isabs(entry.get("image_path")) else entry.get("image_path"),
                "sketch": os.path.abspath(entry.get("conditioning_path")) if not os.path.isabs(entry.get("conditioning_path")) else entry.get("conditioning_path")
            }

            if not all(os.path.exists(p) for p in paths.values()):
                continue

            try:
                # Structure changed to generic "object" key
                entry_results = {
                    "img_bbox_coord": {"object": []},
                    "sketch_bbox_coord": {"object": []},
                    "saved_files": {}
                }
                
                has_any_detection = False 
                
                for key_type in ["img", "sketch"]:
                    current_path = paths[key_type]
                    
                    image_array = load_image(current_path)
                    if image_array is None:
                        continue
                    
                    # --- PREDICTION CHANGE ---
                    # 1. Set image
                    predictor.set_image(image_array)
                    
                    # 2. Predict WITHOUT text prompts
                    # This usually triggers "segment everything" or "detect all" mode
                    results = predictor(text=["object"])
                    
                    grouped_bboxes = {"object": []}

                    if results and results[0].boxes is not None:
                        result = results[0]
                        boxes_data = result.boxes.xyxy.cpu().numpy().tolist()
                        scores_data = result.boxes.conf.cpu().numpy().tolist()
                        
                        # Note: We ignore 'cls' (class_indices) because we are detecting everything
                        # and treating it as a single generic class "object"

                        for box, score in zip(boxes_data, scores_data):
                            if score > CONF_THRESHOLD:
                                grouped_bboxes["object"].append(box)
                    
                    entry_results[f"{key_type}_bbox_coord"] = grouped_bboxes
                    
                    if grouped_bboxes["object"]:
                        has_any_detection = True

                    # 3. Visualization
                    colors = {"img": (0, 255, 0), "sketch": (0, 0, 255)} # Green / Red
                    viz = draw_bboxes(image_array, grouped_bboxes, color=colors[key_type])
                    
                    save_name = f"{item_id}_{key_type}_bbox.jpg"
                    save_abs = os.path.join(dirs[key_type], save_name)
                    cv2.imwrite(save_abs, viz)
                    entry_results["saved_files"][f"{key_type}_bboxed"] = save_abs

                # Logic: Keep entry if at least ONE file had detections? 
                # Or do you require detections in BOTH? 
                # Currently: If ANYTHING is detected in EITHER, we save it.
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
                # import traceback
                # traceback.print_exc()
                continue

    print(f"Done! Processed {valid_count} entries.")
    print(f"Saved to {output_jsonl}")

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    INPUT_FILE = "datasets/fscoco_recaped/recaped.jsonl"
    OUTPUT_FILE = "datasets/fscoco_recaped/scored.jsonl"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    process_dataset(INPUT_FILE, OUTPUT_FILE)