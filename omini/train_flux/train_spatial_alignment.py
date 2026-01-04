import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
from functools import partial
import numpy as np

from PIL import Image, ImageDraw

from datasets import load_dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type: str = "canny",
        drop_text_prob: float = 0,
        drop_image_prob: float = 0,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __get_condition__(self, image, condition_type, pre_loaded_condition=None):
        condition_size = self.condition_size
        position_delta = np.array([0, 0])
        if condition_type in ["canny", "coloring", "deblurring", "depth", "sketch_edit"]:
            image, kwargs = image.resize(condition_size), {}
            if condition_type == "deblurring":
                blur_radius = random.randint(1, 10)
                kwargs["blur_radius"] = blur_radius
            if condition_type == "sketch_edit" and pre_loaded_condition:
                condition_img = pre_loaded_condition.resize(condition_size).convert("RGB")
                # [Optional]: Randomly invert colors if sketches vary between black-on-white vs white-on-black
                # if random.random() > 0.5:
                #     condition_img = Image.eval(condition_img, lambda x: 255 - x)
                return condition_img, position_delta
            condition_img = convert_to_condition(condition_type, image, **kwargs)
        elif condition_type == "depth_pred":
            depth_img = convert_to_condition("depth", image)
            condition_img = image.resize(condition_size)
            image = depth_img.resize(condition_size)
        elif condition_type == "fill":
            condition_img = image.resize(condition_size).convert("RGB")
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
        elif condition_type == "sr":
            condition_img = image.resize(condition_size)
            position_delta = np.array([0, -condition_size[0] // 16])
        else:
            raise ValueError(f"Condition type {condition_type} is not  implemented.")
        return condition_img, position_delta

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        if "image_path" in item:
            # Handle JSONL format (paths can be absolute or relative)
            try:
                image = Image.open(item["image_path"]).convert("RGB")
            except FileNotFoundError:
                # If loading fails (e.g., path is relative), try relative to CWD
                if not os.path.isabs(item["image_path"]):
                    image = Image.open(os.path.join(os.getcwd(), item["image_path"])).convert("RGB")
                else:
                    raise
        elif "jpg" in item:
            # Handle WebDataset format fallback
            image = item["jpg"].convert("RGB")
        else:
            raise KeyError(f"Item {idx} missing 'image_path' or 'jpg' key")
        
        image = image.resize(self.target_size).convert("RGB")
        
        if "caption" in item:
            description = item.get("caption", "")
        elif "json" in item and "prompt" in item["json"]:
            description = item["json"]["prompt"]
        else:
            description = ""

        external_condition = None
        if "conditioning_path" in item:
            cond_path = item["conditioning_path"]
            if os.path.exists(cond_path):
                external_condition = Image.open(cond_path).convert("RGB")
            elif not os.path.isabs(cond_path):
                 # Try relative path
                 rel_path = os.path.join(os.getcwd(), cond_path)
                 if os.path.exists(rel_path):
                     external_condition = Image.open(rel_path).convert("RGB")

        condition_size = self.condition_size
        position_scale = self.position_scale

        condition_img, position_delta = self.__get_condition__(
            image, self.condition_type, external_condition
        )

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new("RGB", condition_size, (0, 0, 0))

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": position_delta,
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }


@torch.no_grad()
def test_function(model, save_path, file_name, validation_samples=None):
    """
    [Modified]: Accepts a list/dataset of validation_samples.
    Iterates through all provided samples to generate validation images.
    Saves Original (GT), Condition (Sketch), and Generated Result.
    """
    # Extract config
    train_config = model.training_config
    condition_size = train_config["dataset"]["condition_size"]
    target_size = train_config["dataset"]["target_size"]
    position_delta = train_config["dataset"].get("position_delta", [0, 0])
    position_scale = train_config["dataset"].get("position_scale", 1.0)
    adapter = model.adapter_names[2]
    condition_type = train_config["condition_type"]
    
    # List to hold: (ConditionObject, Prompt, FilenameSuffix, OriginalImage_PIL, ConditionImage_PIL)
    test_items = []

    # 1. Prepare Inputs
    if validation_samples is not None and len(validation_samples) > 0:
        print(f"Preparing validation for {len(validation_samples)} samples...")
        
        for idx, sample in enumerate(validation_samples):
            # Load Original Image (GT)
            img_p = sample.get("image_path")
            suffix = os.path.basename(img_p).split('.')[0]
            if not os.path.isabs(img_p): img_p = os.path.join(os.getcwd(), img_p)
            # Keep original RGB for saving
            original_image = Image.open(img_p).convert("RGB").resize(condition_size)
            
            prompt = sample.get("caption", "")
            
            # Load Pre-existing Condition (Sketch) if available
            pre_cond_path = sample.get("conditioning_path")
            condition_img_vis = None # This will store the sketch/mask for visualization
            
            if pre_cond_path:
                if not os.path.isabs(pre_cond_path): pre_cond_path = os.path.join(os.getcwd(), pre_cond_path)
                if os.path.exists(pre_cond_path):
                    condition_img_vis = Image.open(pre_cond_path).convert("RGB").resize(condition_size)

            # Determine Condition Logic per sample
            cond_obj = None
            
            if condition_type == "sketch_edit" and condition_img_vis is not None:
                # Use the pre-loaded sketch
                cond_obj = Condition(condition_img_vis, adapter, position_delta, position_scale)
                
            elif condition_type in ["canny", "coloring", "deblurring", "depth", "sketch_edit"]:
                # On-the-fly generation
                blur = 5 if condition_type == "deblurring" else None
                # If we computed it on the fly, update the visualization image
                condition_img_vis = convert_to_condition(condition_type, original_image, blur_radius=blur)
                cond_obj = Condition(condition_img_vis, adapter, position_delta, position_scale)
                
            elif condition_type == "fill":
                # Create mask
                w, h = original_image.size
                mask = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([w//4, h//4, w*3//4, h*3//4], fill=255)
                # Create masked image
                masked_img = Image.composite(original_image, Image.new("RGB", (w, h), 0), Image.eval(mask, lambda a: 255 - a))
                condition_img_vis = masked_img # For viz, show the masked input
                cond_obj = Condition(masked_img, adapter, position_delta, position_scale)
                
            else:
                # Default
                cond_obj = Condition(original_image, adapter, position_delta, position_scale)
                condition_img_vis = original_image

            test_items.append((cond_obj, prompt, suffix, original_image, condition_img_vis))

    else:
        # [Modified]: Fallback Asset Test
        print("Validation list empty. Using default asset.")
        img_p = "assets/doggy_tree.png"
        if os.path.exists(img_p):
            original_image = Image.open(img_p).convert("RGB").resize(condition_size)
        else:
            # Create a dummy image if asset missing
            original_image = Image.new("RGB", condition_size, (128, 128, 128))
            
        cond_obj = Condition(original_image, adapter, position_delta, position_scale) 
        test_items.append((cond_obj, "Two dogs under a big tree.", "default_asset", original_image, original_image))

    # 2. Generation Loop
    save_dir = os.path.join(save_path, 'test')
    os.makedirs(save_dir, exist_ok=True)
    
    for _, (condition, prompt, suffix, orig_img, cond_img) in enumerate(test_items):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
        )
        
        generated_img = res.images[0]

        # [Modified]: Save Original, Condition, and Result
        
        # 1. Save Result
        gen_path = os.path.join(save_dir, f"{file_name}_{condition_type}_{suffix}_gen.jpg")
        generated_img.save(gen_path)
        
        # 2. Save Original (GT)
        orig_path = os.path.join(save_dir, f"{file_name}_{condition_type}_{suffix}_orig.jpg")
        orig_img.save(orig_path)
        
        # 3. Save Condition (Sketch/Mask)
        if cond_img:
            cond_path = os.path.join(save_dir, f"{file_name}_{condition_type}_{suffix}_cond.jpg")
            cond_img.save(cond_path)
        
        # 4. (Optional) Save a Grid for easy comparison: [Original | Condition | Generated]
        w, h = generated_img.size
        grid = Image.new("RGB", (w * 3, h))
        grid.paste(orig_img.resize((w, h)), (0, 0))
        if cond_img:
            grid.paste(cond_img.resize((w, h)), (w, 0))
        grid.paste(generated_img, (w * 2, 0))
        
        grid_path = os.path.join(save_dir, f"{file_name}_{condition_type}_{suffix}_grid.jpg")
        grid.save(grid_path)
        
        print(f"Saved validation sample to {save_dir} (suffix: {suffix})")

def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0))) 

    # 1. Load the full dataset
    raw_dataset = load_dataset(
        "json",
        data_files={"train": training_config["dataset"]["urls"]},
        split="train",
        cache_dir="datasets/fscoco",
    )
    
    print(f"Total samples loaded: {len(raw_dataset)}")

    # 2. Perform Train-Test Split (e.g., 99% train, 1% test)
    # Using a fixed seed ensures we always validate on the same set
    split_dataset = raw_dataset.train_test_split(test_size=training_config["dataset"]["test_ratio"], seed=42)
    train_split = split_dataset["train"]
    test_split = split_dataset["test"]

    # Initialize custom dataset
    dataset = ImageConditionDataset(
        train_split,
        condition_size=training_config["dataset"]["condition_size"],
        target_size=training_config["dataset"]["target_size"],
        condition_type=training_config["condition_type"],
        drop_text_prob=training_config["dataset"]["drop_text_prob"],
        drop_image_prob=training_config["dataset"]["drop_image_prob"],
        position_scale=training_config["dataset"].get("position_scale", 1.0),
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    test_function_with_samples = partial(test_function, validation_samples=test_split)
    train(dataset, trainable_model, config, test_function_with_samples)

if __name__ == "__main__":
    main()
