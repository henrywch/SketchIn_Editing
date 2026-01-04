import sys
import os
import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline

from ..pipeline.flux_omini import Condition, convert_to_condition, generate

def load_trained_model_with_lora(model_path, checkpoint_path):
    """Load FLUX-1-dev with trained LoRA weights"""
    print(f"Loading base model from: {model_path}")
    
    # 1. Load base FLUX-1-dev model
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # 2. Clear any existing LoRAs (Best Practice from spatial.ipynb)
    pipe.unload_lora_weights()

    # 3. Load your trained LoRA weights
    print(f"Loading LoRA adapter from: {checkpoint_path}")
    
    pipe.load_lora_weights(
        checkpoint_path,
        weight_name="default.safetensors",
        adapter_name="spatial"
    )

    # 4. Set the adapter as active
    pipe.set_adapters(["spatial"], adapter_weights=[1.0])
    
    return pipe

def inference_spatial_alignment(pipe, image_path, prompt, condition_type="sketch-edit", condition_path=None):
    """Run inference with spatial alignment control"""
    condition_size = [512, 512]
    target_size = [512, 512]
    
    # Load and prepare input image (Source Image)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    image = Image.open(image_path).resize(condition_size).convert("RGB")

    # Generate condition based on type
    if condition_type == "sketch-edit":
        if condition_path and os.path.exists(condition_path):
            print(f"Loading custom condition from: {condition_path}")
            condition_img = Image.open(condition_path).resize(condition_size).convert("RGB")
        else:
            print("Generating condition from input image...")
            condition_img = convert_to_condition(condition_type, image)
    else:
        # Fallback for other types supported by official repo
        condition_img = convert_to_condition(condition_type, image, blur_radius=5)

    # Create condition object
    # Matches spatial.ipynb: Condition(image, type_name)
    # Ensure 'type_name' here matches the 'adapter_name' loaded earlier ("spatial")
    condition = Condition(
        condition_img, 
        "spatial", 
        np.array([0, 0]), 
        1.0 
    )

    print("Running generation...")
    with torch.no_grad():
        result = generate(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=torch.Generator(device=pipe.device).manual_seed(42),
            model_config={},
            kv_cache=True,
        )

    return result.images[0]

if __name__ == "__main__":
    # Define paths
    MODEL_PATH = "models/FLUX.1-dev"
    CHECKPOINT_DIR = "checkpoints/head.safetensors"
    # CHECKPOINT_DIR = "checkpoints/character.safetensors"
    INPUT_IMAGE = "samples/heads/430100/430100_orig.png"
    CONDITION_IMAGE = "samples/heads/430100/430100_combined.jpg"
    # INPUT_IMAGE = "samples/characters/378013/378013_orig.png"
    # CONDITION_IMAGE = "samples/characters/378013/378013_combined.jpg"
    
    # Load model
    pipe = load_trained_model_with_lora(MODEL_PATH, CHECKPOINT_DIR)
    
    # Run inference
    output_image = inference_spatial_alignment(
        pipe=pipe,
        image_path=INPUT_IMAGE,
        prompt="Please fill in the image according to the sketch in the image",
        condition_type="sketch-edit",
        condition_path=CONDITION_IMAGE
    )
    
    # Save result
    output_path = "output.jpg"
    output_image.save(output_path)
    print(f"Result saved to {output_path}")