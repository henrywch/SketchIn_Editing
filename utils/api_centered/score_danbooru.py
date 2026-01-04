"""
Use PYIQA Metrics to score the original Danbooru images and sketches
"""

import os
import json
import torch
import argparse
import math
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pyiqa

# ================= CONFIGURATION =================
# Batch size per GPU (RTX 4090 24GB can handle 32-64 easily)
BATCH_SIZE = 32 
NUM_WORKERS = 4  # CPU threads per GPU to load images
IMAGE_SIZE = 512 # Resize images to this to allow batching

METRICS_MAP = {
    "musiq": "musiq",       
    "nima": "nima",         
    "maniqa": "maniqa",     
    "hyperiqa": "hyperiqa", 
    "clipiqa": "clipiqa",   
    "niqe": "niqe"          
}

class SketchDataset(Dataset):
    def __init__(self, file_list, img_size):
        self.files = file_list
        # Standard transform: Resize -> Tensor (0-1 range)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor() 
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = str(self.files[idx])
        try:
            # Convert to RGB to ensure 3 channels
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
            return path, tensor
        except Exception as e:
            # Return None to handle in collate_fn
            return path, None

def collate_fn(batch):
    """Filter out broken images (None) so the batch doesn't crash"""
    batch = list(filter(lambda x: x[1] is not None, batch))
    if not batch:
        return [], None
    paths, tensors = zip(*batch)
    return paths, torch.stack(tensors)

def worker(gpu_id, image_files, output_file):
    device = torch.device(f"cuda:{gpu_id}")
    
    # 1. Load Models
    models = {}
    try:
        for key, metric_name in METRICS_MAP.items():
            # Create metric
            model = pyiqa.create_metric(metric_name, device=device)
            model.eval()
            models[key] = model
    except Exception as e:
        print(f"[GPU {gpu_id}] Error loading models: {e}")
        return

    # 2. Create Dataset & Loader
    dataset = SketchDataset(image_files, IMAGE_SIZE)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    results = []
    
    # 3. Batch Inference Loop
    # position logic keeps bars separate
    for paths, batch_tensor in tqdm(loader, desc=f"GPU {gpu_id}", position=gpu_id):
        if batch_tensor is None: continue
        
        batch_tensor = batch_tensor.to(device)
        
        # Prepare a dict for this batch: {image_path: {scores...}}
        batch_results = {p: {"image_path": p} for p in paths}

        with torch.no_grad():
            for key, model in models.items():
                try:
                    # Pass the whole batch tensor [B, 3, H, W]
                    if key == "niqe":
                        # NIQE often crashes on batches or returns 1 value for the whole batch
                        # Safer to iterate manually for just this metric
                        scores = []
                        for i in range(batch_tensor.shape[0]):
                            # NIQE expects 5D input [1, 3, H, W] or 4D
                            s = model(batch_tensor[i].unsqueeze(0))
                            scores.append(float(s.item()))
                    else:
                        # Deep learning models (MUSIQ, NIMA, etc) handle batches natively
                        output = model(batch_tensor)
                        # Ensure output is a list of floats
                        if output.dim() == 0: # Scalar return (bad batching support)
                            scores = [float(output.item())] * len(paths)
                        else:
                            scores = output.cpu().squeeze().tolist()
                            # Handle single-item batch edge case (returns float not list)
                            if isinstance(scores, float):
                                scores = [scores]

                    # Assign scores back to paths
                    for i, p in enumerate(paths):
                        batch_results[p][key] = scores[i]

                except Exception as e:
                    # Fallback: if batching fails for a specific model, fill with -1
                    # print(f"Batch fail {key}: {e}") 
                    for p in paths:
                        batch_results[p][key] = -1.0

        # Flatten dict to list of strings
        for p in paths:
            results.append(json.dumps(batch_results[p]))

        # Write to disk every 10 batches to save RAM
        if len(results) >= (BATCH_SIZE * 10):
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(results) + '\n')
            results = []

    # Final write
    if results:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(results) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="quality_results_batched")
    args = parser.parse_args()

    # 1. Scan Files
    print(f"Scanning images in {args.input_dir}...")
    root_path = Path(args.input_dir)
    extensions = {'*.png', '*.jpg', '*.jpeg', '*.webp'}
    image_files = []
    for ext in extensions:
        image_files.extend(list(root_path.rglob(ext)))
    
    image_files.sort()
    total_images = len(image_files)
    print(f"Found {total_images} images.")

    # 2. Split Data
    num_gpus = torch.cuda.device_count()
    chunk_size = math.ceil(total_images / num_gpus)
    chunks = [image_files[i:i + chunk_size] for i in range(0, total_images, chunk_size)]

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Launch Workers
    mp.set_start_method('spawn', force=True)
    processes = []

    for rank in range(num_gpus):
        worker_output = os.path.join(args.output_dir, f"results_gpu_{rank}.jsonl")
        # clear file
        with open(worker_output, 'w') as f: pass

        if rank < len(chunks):
            p = mp.Process(target=worker, args=(rank, chunks[rank], worker_output))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    print("Merging results...")
    final_output = os.path.join(args.output_dir, "all_scores.jsonl")
    with open(final_output, 'w', encoding='utf-8') as outfile:
        for rank in range(num_gpus):
            worker_output = os.path.join(args.output_dir, f"results_gpu_{rank}.jsonl")
            if os.path.exists(worker_output):
                with open(worker_output, 'r', encoding='utf-8') as infile:
                    # Use shutil for faster large file merge
                    import shutil
                    shutil.copyfileobj(infile, outfile)
    
    print(f"Done! Saved to {final_output}")

if __name__ == "__main__":
    main()