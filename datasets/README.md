### Danbooru Inpainting-Sketch Datasets

> Huggingface: https://huggingface.co/datasets/henrywch2huggingface/Danbooru-Inpainting-Sketch

We filtered Danbooru, a anime datasets with colored and black & white images, with PYIQA metrics (50% filtered for each metric)

![Quality Distribution](quality_distribution.png "PYIQA Score Distribution")

| Metrics | Mean | Thresh. | Expl. |
| --- | --- | --- | --- |
| **MUSIQ** | 28.13 | 27.89 | Evaluates **technical image quality** (sharpness, granularity, distortion) across multiple scales and resolutions using a Transformer-based architecture. |
| **NIMA** | 3.67 | 3.66 | Focuses on **subjective aesthetics** and artistic composition, predicting how a human would rate the image's beauty rather than just its technical fidelity. |
| **MANIQA** | 0.24 | *(Discard)\** | Assesses **perceptual quality** using attention mechanisms to identify and weigh interaction between local detailed regions (patches) and the global image. |
| **HYPERIQA** | 0.25 | 0.24 | A content-aware metric that evaluates **perceptual quality** by adapting its network parameters to the specific visual content (e.g., distinguishing between texture and noise). |
| **CLIPIQA** | 0.37 | 0.37 | Measures **semantic quality** by using the CLIP model to calculate how closely the image matches the text concept of "Good photo" versus "Bad photo". |
| **NIQE** | 11.02 | 11.09 | Measures **statistical naturalness**. It calculates how much the image deviates from the statistical regularities observed in pristine, natural images (focusing on artifacts). |

*\* as only few images succeed on maniqa score, we excluded this metric when filtering.*

And call SAM3 to detect the heads and characters in the images and sketches. Then we replace the head/character bbox on the original image (the highest confidence) with the head/character bbox on the sketch (the highest confidence) to form the sketch-embbeded anime image datasets.

The parquet files are keyed with `{id, uid, image (binary), condition (the embbeded image, binary), image_path, conditioning_path, caption}`. For **finetuning quickstart**, you only need to load the images and conditions and save them to the image_path and conditioning_path, respectively, then you can save a metadata file as *jsonl* formatting `{"image_path": "", "conditioning_path": "", "caption": ""}`, and you'll be on the fly.