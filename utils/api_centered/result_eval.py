"""
Evaluate the Danbooru Inpainting Sketch Training Results
"""

import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from torchmetrics.image.fid import FrechetInceptionDistance
import pyiqa
import os
import matplotlib.pyplot as plt

METRICS_MAP = {
    "musiq": "musiq",       
    "nima": "nima",         
    "maniqa": "maniqa",     
    "hyperiqa": "hyperiqa", 
    "clipiqa": "clipiqa",   
    "niqe": "niqe"          
}

def load_image_for_metric(path, as_tensor=False, for_pyiqa=False, device='cpu'):
    """
    Loads an image. 
    - standard: Returns Numpy array (H, W, 3) [0-255]
    - as_tensor (FID): Returns Torch Tensor (1, 3, H, W) [0-255] uint8
    - for_pyiqa: Returns Torch Tensor (1, 3, H, W) [0-1] float32
    """
    try:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        
        if for_pyiqa:
            # PyIQA usually expects Float Tensor [0, 1]
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            return img_tensor.to(device)

        if as_tensor:
            # TorchMetrics FID expects [N, 3, H, W] uint8
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.uint8)
            return img_tensor.to(device)
            
        return img_np
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def plot_metrics(results, metric_keys, title, output_filename):
    """
    Generates a figure with subplots for each metric.
    """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    num_metrics = len(metric_keys)
    cols = 2
    rows = (num_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    # Extract IDs for X-axis
    ids = [r['id'] for r in results]
    x = range(len(ids))

    for i, metric in enumerate(metric_keys):
        # Extract values for this metric
        # Check if the metric exists in the 'scores' dictionary (it might be in 'extra_scores')
        values = []
        for r in results:
            if metric in r['scores']:
                values.append(r['scores'][metric])
            elif metric in r.get('extra_scores', {}):
                values.append(r['extra_scores'][metric])
            else:
                values.append(0) # Fallback

        ax = axes[i]
        ax.plot(x, values, marker='o', linestyle='-', markersize=4, label=metric)
        ax.set_title(metric.upper())
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate mean for reference line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.legend()

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Saving plot to {output_filename}")
    plt.savefig(output_filename)
    plt.close()

def evaluate_and_save(input_jsonl, output_jsonl):
    print(f"Reading from: {input_jsonl}")
    print(f"Saving to:    {output_jsonl}")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Metrics
    # A. FID
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    
    # B. PyIQA Metrics
    iqa_metrics = {}
    print("Initializing IQA metrics (this may download weights)...")
    for name, metric_name in METRICS_MAP.items():
        try:
            # create_metric returns a torch module
            iqa_metrics[name] = pyiqa.create_metric(metric_name, device=device)
            print(f" -> Loaded {name}")
        except Exception as e:
            print(f" -> Failed to load {name}: {e}")

    # 3. Load Input Data
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results_buffer = []
    
    # 4. Process Loop
    print("Processing images...")
    for line in tqdm(lines, desc="Evaluating", unit="pair"):
        data = json.loads(line)
        
        record_id = data.get("id")
        real_path = data.get("orig")
        gen_path = data.get("gen")

        if not os.path.exists(real_path) or not os.path.exists(gen_path):
            continue

        # --- Load Images ---
        # 1. Numpy for PSNR/SSIM
        real_np = load_image_for_metric(real_path, as_tensor=False)
        gen_np = load_image_for_metric(gen_path, as_tensor=False)

        # 2. Tensor for PyIQA (0-1 float)
        real_tensor_iqa = load_image_for_metric(real_path, for_pyiqa=True, device=device)
        gen_tensor_iqa = load_image_for_metric(gen_path, for_pyiqa=True, device=device)

        # Handle shape mismatch for SSIM
        if real_np.shape != gen_np.shape:
            img_gen_pil = Image.fromarray(gen_np).resize((real_np.shape[1], real_np.shape[0]))
            gen_np = np.array(img_gen_pil)

        # --- A. Standard Metrics (PSNR/SSIM) ---
        p_val = psnr_func(real_np, gen_np, data_range=255)
        s_val = ssim_func(real_np, gen_np, data_range=255, channel_axis=-1, win_size=3)

        # --- B. FID Update ---
        real_tensor_fid = torch.from_numpy(real_np).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.uint8).to(device)
        gen_tensor_fid = torch.from_numpy(gen_np).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.uint8).to(device)
        fid_metric.update(real_tensor_fid, real=True)
        fid_metric.update(gen_tensor_fid, real=False)

        # --- C. IQA Metrics (Original vs Generated) ---
        current_scores_real = {}
        current_scores_gen = {}
        
        with torch.no_grad():
            for name, model in iqa_metrics.items():
                # Some metrics return a scalar tensor, use .item()
                try:
                    score_real = model(real_tensor_iqa).item()
                    score_gen = model(gen_tensor_iqa).item()
                    current_scores_real[name] = score_real
                    current_scores_gen[name] = score_gen
                except Exception as e:
                    print(f"Error running {name}: {e}")
                    current_scores_real[name] = 0.0
                    current_scores_gen[name] = 0.0

        # Store results
        # We segregate metrics by "Original Image Scores" and "Generated Image Scores"
        results_buffer.append({
            "id": record_id,
            "image_path": real_path,
            "generated_path": gen_path,
            "scores": {
                "psnr": float(p_val),
                "ssim": float(s_val)
            },
            "metrics_original": current_scores_real,
            "metrics_generated": current_scores_gen
        })

    # 5. Compute Global FID
    print("Calculating final global FID...")
    final_fid = fid_metric.compute().item()

    # 6. Write to Output JSONL
    print(f"Writing results to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for record in results_buffer:
            output_record = {
                "id": record["id"],
                "paths": {
                    "orig": record["image_path"],
                    "gen": record["generated_path"]
                },
                "pair_metrics": {
                    "psnr": record["scores"]["psnr"],
                    "ssim": record["scores"]["ssim"],
                    "fid": final_fid
                },
                "original_image_quality": record["metrics_original"],
                "generated_image_quality": record["metrics_generated"]
            }
            f_out.write(json.dumps(output_record) + "\n")

    # 7. Generate Plots
    # Prepare data for plotting
    # We want two plots: 
    #   1. Original Images (X=ID, Y=Metric) for all IQA metrics
    #   2. Generated Images (X=ID, Y=Metric) for all IQA metrics
    
    # Flatten structure for the plotter
    plot_data_orig = []
    plot_data_gen = []
    
    for r in results_buffer:
        plot_data_orig.append({"id": r["id"], "extra_scores": r["metrics_original"], "scores": {}})
        plot_data_gen.append({"id": r["id"], "extra_scores": r["metrics_generated"], "scores": r["scores"]}) # Gen also has PSNR/SSIM

    # Define keys to plot
    iqa_keys = list(METRICS_MAP.keys())
    
    # Plot 1: Original Images (Only IQA metrics apply)
    output_dir = os.path.dirname(output_jsonl)
    plot_metrics(plot_data_orig, iqa_keys, "Quality Metrics - Original Images", os.path.join(output_dir, "metrics_plot_original.png"))
    
    # Plot 2: Generated Images (IQA metrics + PSNR/SSIM)
    # We add PSNR/SSIM to the list for generated images
    gen_keys = ["psnr", "ssim"] + iqa_keys
    plot_metrics(plot_data_gen, gen_keys, "Quality Metrics - Generated Images", os.path.join(output_dir, "metrics_plot_generated.png"))

    print("\n" + "="*30)
    print("       EVALUATION DONE       ")
    print("="*30)
    print(f"Global FID:   {final_fid:.4f}")
    print("Plots saved to output directory.")

if __name__ == "__main__":
    # Update paths as per your request
    INPUT_JSONL = "datasets/danbooru/half/replaced/head/metadata.jsonl"
    OUTPUT_JSONL = "datasets/danbooru/half/replaced/head/results.jsonl"
    
    if os.path.exists(INPUT_JSONL):
        evaluate_and_save(INPUT_JSONL, OUTPUT_JSONL)
    else:
        print(f"File not found: {INPUT_JSONL}")