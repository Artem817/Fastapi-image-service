# Image Processing Service

![Status: Proof of Concept](https://img.shields.io/badge/Status-Proof_of_Concept-yellow)

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)


![Python](https://img.shields.io/badge/python-3.11-blue)


Simple FastAPI service for uploading images and applying basic transformations
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
2. Provide environment variables (see below).
3. Run the API:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`

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

## Background Removal Architecture
The background removal model is a U-Net style decoder on top of a ResNet-101 encoder.
It uses ResNet-101 blocks for feature extraction and a multi-stage decoder with
upsampling, skip connections, and Conv/BN/ReLU blocks to produce a 1-channel mask.

Implementation details:
- Encoder: `torchvision.models.resnet101` layers up to `layer4`.
- Decoder: stacked `DecoderBlock` stages with bilinear upsample and skip concatenation.
- Head: `1x1` conv to get a single-channel logits mask.
- Inference: logits are resized to input size and passed through `sigmoid` to get the mask.

Weights are loaded from `app/models_unet/resnet101_unet.pth` at import time in
`app/models_unet/model_arch.py`. If the file is missing, the remove-bg endpoint
will not work.


## Core Endpoints
- `POST /images/upload`
- `POST /images/filter`
- `POST /images/resize`
- `POST /images/rotate`
- `POST /images/flip`
- `POST /images/watermark`
- `DELETE /images/{image_id}`
