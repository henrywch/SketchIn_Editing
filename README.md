## SketchIn-Editing

> Github: https://github.com/henrywch/SketchIn_Editing.git

### Table of Contents
- [Introduction](#introduction)
- [Paper](#paper)
- [Quickstart](#quickstart)
- [Datasets](#datasets)

### Introduction

We deploy the OminiControl Training Framework ([Tan et al., 2025](https://arxiv.org/html/2411.15098v3 "paper link")) to do LoRA Finetuning on FLUX-1.dev model, with a customized dataset Danbooru-InPainting-Sketch (DIPS).

### Paper

Please refer to [Efficient Sketch-Guided Image Inpainting via Composite Condition and Matched RoPE](paper/Efficient_Sketch-Guided_Image_Inpainting_via_Composite_Condition_and_Matched_RoPE.pdf "Inpainting Sketch Editing Paper")

### Quickstart

1. Build the environment (feel free to transfer the yaml files to `requirements.txt` if you'd like to use python virtual envs)

```bash
conda env create -f sie.yaml
conda activate sie
```

2. Download the dataset from huggingface.
(Details refer to [Datasets](#datasets), or you can directly access [datasets/README.md](datasets/README.md "Dataset README.md") as you'll be similarly told down there)
```bash
hf auth login 
```

3. Modify the comfigurations.
(Common parts shown below, details refering to [spatial_alignment.sh](train/config/spatial_alignment.yaml "training script"))
```yaml
flux_path: "(your model path)"
dtype: "bfloat16"

model:
  independent_condition: true

train:
  accumulate_grad_batches: 1
  dataloader_workers: 5
  save_interval: 1000
  sample_interval: 100
  max_steps: -1
  gradient_checkpointing: true # (Turn off for faster training)
  save_path: "runs"

  # Specify the type of condition to use. 
  # Options: ["sketch_edit", "canny", "coloring", "deblurring", "depth", "depth_pred", "fill"]
  condition_type: "sketch_edit"
  dataset:
    type: "img"
    urls:
      - "datasets/dips_head.jsonl"
    cache_name: "dips_head"
    condition_size: 
      - 512
      - 512
    target_size: 
      - 512
      - 512
    drop_text_prob: 0
    drop_image_prob: 0
    test_ratio: 0.05

  lora_config:
    r: 16
    lora_alpha: 16
    ...

  optimizer:
    type: "Prodigy"
    params:
      lr: 1.0
      use_bias_correction: true
      safeguard_warmup: true
      weight_decay: 0.01
```

4. Start training

```bash
bash train/script/train_spatial_alignment.sh
```

5. Start Infering

```bash
python -m omini.infer_flux.inference
```

### Datasets

Please refer to [datasets/README.md](datasets/README.md "Dataset README.md")
