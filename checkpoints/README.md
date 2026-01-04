### OminiControl LoRA Checkpoints

> Huggingface: https://huggingface.co/henrywch2huggingface/inpainting-sketch-editing

We adapted the **OminiContrlol Framework** to train our **sketch-conditioned inpainting model** with **LoRA finetuning**. We used the *head-sketched* and *character-sketched* datasets to train 2 checkpoints, and achieveing final losses of **0.15** and **0.18** respectively.

| Checkpoint | Dataset | Epoch | Loss |
| --- | --- | --- | --- |
| head_sketched.safetensors | *danbooru_inpainting_sketch_head* | 5 | **0.15** |
| character_sketched.safetensors | *danbooru_inpainting_sketch_head* | 10 | **0.18** |