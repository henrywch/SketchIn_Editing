import torch
import os
import argparse
from datasets import load_dataset
from PIL import Image

# Import your existing modules
# Adjust relative imports based on your folder structure
from .trainer import OminiModel, get_config
from .train_spatial_alignment import test_function  # The function we wrote previously

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="runs/20251212-142607/ckpt/4000", help="Path to the saved LoRA folder (e.g., runs/ckpt/5000)")
    parser.add_argument("--output_dir", type=str, default="output", help="Where to save results")
    args = parser.parse_args()

    # 1. Setup Config & Device
    config = get_config("train/config/spatial_alignment.yaml")
    training_config = config["train"]
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device_id)
    device = "cuda"

    print(f"Initializing Base FLUX Model from {config['flux_path']}...")
    
    # 2. Initialize Model (This loads FLUX base)
    # We use the exact same initialization as training to ensure architecture matches
    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=device,
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
    )

    # 3. Load Trained LoRA Weights
    print(f"Loading LoRA weights from {args.ckpt_path}...")

    model.training_config = training_config
    
    # OminiModel usually has a load_lora method corresponding to save_lora.
    # If not, it typically uses PEFT's load_adapter.
    # Assuming 'load_lora' exists based on your snippet 'pl_module.save_lora':
    if hasattr(model, "load_lora"):
        model.load_lora(args.ckpt_path)
    else:
        # Fallback: Standard PEFT loading
        model.flux_pipe.load_lora_weights(args.ckpt_path)

    # Switch to Eval mode (disables dropout, etc.)
    model.eval()

    # 4. Prepare Validation Data
    # We load the dataset just to get the test split, exactly like in training
    print("Loading Dataset for Validation...")
    raw_dataset = load_dataset(
        "json",
        data_files={"train": training_config["dataset"]["urls"]},
        split="train",
        cache_dir="datasets/fscoco",
    )
    
    # Re-create the split to ensure we get the SAME test set as training
    split_dataset = raw_dataset.train_test_split(test_size=0.01, seed=42)
    test_split = split_dataset["test"]
    
    print(f"Found {len(test_split)} test samples.")

    # 5. Run Inference
    # We pass the 'model' here, which acts as the 'pl_module'
    print("Running Test Function...")
    test_function(
        model=model,
        save_path=args.output_dir,
        file_name="inference",
        validation_samples=test_split # Pass all test samples
    )

    print(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()