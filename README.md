# Image Processing Service

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)


FastAPI service for uploading images and applying basic transformations
(filter, resize, rotate, flip, watermark). Images are cached in Redis and
metadata is stored in PostgreSQL via SQLAlchemy.

Background removal is supported only for people and produces visible artifacts.
Quality is not comparable to top-tier tools yet, but it will improve.

## Tech Stack
- FastAPI
- Redis
- PostgreSQL + SQLAlchemy
- Pillow

## Quick Start
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and update values.
3. Provide environment variables (see below).
4. Run the API:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Recent Updates
- Added domain-level exceptions with a centralized FastAPI handler.
- Hardened upload validation (size cap and file signature checks).
- Fixed Redis cleanup for image deletion and improved not-found responses.
- Added async upload tests with mocked Redis and HTTPX client fixtures.
- Modernized SQLAlchemy base declaration and refreshed dependencies.

## Environment Variables
- `DATABASE_URL` (required)
- `SECRET_KEY` (required)
- `REDIS_URL` (optional, default `redis://localhost:6379`)
- `MODEL_PATH` (optional, absolute path to `.pth` weights)
- `MODEL_URL` (optional, download URL for `.pth` weights)

## Model Weights (.pth)
In production, weights are typically stored outside the repo (S3/GCS/Artifacts)
and downloaded at deploy time. For small projects you can use Git LFS, but avoid
committing large `.pth` files directly to git.

Recommended approach:
- Keep weights in a `models/` directory ignored by git.
- Provide `MODEL_PATH` to use a local file.
- Or provide `MODEL_URL` and download the file at startup or via a setup script.
  A GitHub Release asset URL or S3/GCS URL works well here.

## Model: U-Net (ResNet-101)
The background removal model is a U-Net style decoder on top of a ResNet-101 encoder.
It uses ResNet-101 blocks for feature extraction and a multi-stage decoder with
upsampling, skip connections, and Conv/BN/ReLU blocks to produce a 1-channel mask.

Implementation details:
- Encoder: `torchvision.models.resnet101` layers up to `layer4`.
- Decoder: stacked `DecoderBlock` stages with bilinear upsample and skip concatenation.
- Head: `1x1` conv to get a single-channel logits mask.
- Inference: logits are resized to input size and passed through `sigmoid` to get the mask.

Weights are loaded in `app/models_unet/model_arch.py` from `MODEL_PATH` or downloaded
from `MODEL_URL` (or a local `app/models_unet/resnet101_unet.pth` if present). If
weights are missing, the remove-bg endpoint returns an error.


###  Technical Specifications
* **Architecture:** U-Net with ResNet-101 Backbone.
* **Training Hardware:** Kaggle P100 GPU.
* **Training Duration:** 10 Epochs.
* **Key Strengths:** High-fidelity edge detection, especially in complex areas like facial hair, headwear, and fine textures.

### Results Demonstration

| Input vs. Output Segmentation |
| :---: |
| <img src="https://github.com/user-attachments/assets/f49f74b9-fbf6-41d6-828f-251a12496452" width="600" alt="U-Net Result"> |
| *Left: Original Image | Right: Processed result showing clean, precise edges.* |

> [!TIP]
> You may notice a slight halo around complex contours (hair, cap edges).The model performs semantic segmentation (classification of pixels as "person/background") rather than Alpha Matting (calculation of edge transparency).

*Photo by [X-Outcast](https://unsplash.com/@xoutcastx) via **Unsplash**.*

## Core Endpoints
- `POST /images/upload`
- `POST /images/filter`
- `POST /images/resize`
- `POST /images/rotate`
- `POST /images/flip`
- `POST /images/watermark`
- `DELETE /images/{image_id}`
