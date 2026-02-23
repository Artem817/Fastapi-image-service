import os
import urllib.request
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

def ensure_model_weights():
    """
    Locate model weights (.pth). Order of precedence:
    1. Explicit path from ENV `MODEL_PATH`
    2. Local bundled file `resnet101_unet.pth`
    3. Download from ENV `MODEL_URL` (falls back to GitHub release URL)
    """

    env_path = os.getenv("MODEL_PATH")
    if env_path:
        if os.path.exists(env_path):
            print(f"✓ Using model weights from MODEL_PATH: {env_path}")
            return env_path
        print(f"✗ MODEL_PATH is set but file not found: {env_path}")

    model_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(model_dir, "resnet101_unet.pth")

    if os.path.exists(weights_path):
        print(f"✓ Model weights found at: {weights_path}")
        return weights_path

    download_url = os.getenv(
        "MODEL_URL",
        "https://github.com/Artem817/Fastapi-image-service/releases/download/v1.0.0-rc1/resnet101_unet.pth",
    )

    if not download_url:
        print("✗ MODEL_URL not provided and local weights missing.")
        return None

    print("Model weights not found locally. Attempting to download...")
    try:
        print(f"   Downloading from: {download_url}")
        urllib.request.urlretrieve(download_url, weights_path)
        print(f"✓ Download complete! Saved to: {weights_path}")
        return weights_path
    except urllib.error.URLError as e:
        print(f"✗ Failed to download weights: {e}")
        print(f"   Please provide MODEL_PATH or place the file at: {weights_path}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error during download: {e}")
        return None

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet101Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet101(weights=None)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.pool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x0, x2, x3, x4, x5


class ResNet101Unifier(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.encoder = ResNet101Encoder()
        self.decoder4 = DecoderBlock(2048, 1024, 1024)
        self.decoder3 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder1 = DecoderBlock(256, 64, 128)
        self.decoder0 = DecoderBlock(128, 0, 64)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:]
        x0, x2, x3, x4, x5 = self.encoder(x)
        d4 = self.decoder4(x5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        d1 = self.decoder1(d2, x0)
        d0 = self.decoder0(d1, None)
        logits = self.out(d0)
        if logits.shape[2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

model_ready = False
model = ResNet101Unifier(n_classes=1).to(device)

weights_path = ensure_model_weights()

if weights_path and os.path.exists(weights_path):
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model_ready = True
        print("Model loaded successfully and set to eval mode")
    except Exception as e:
        model_ready = False
        print(f"Error loading model weights: {e}")
        print(f"Make sure {weights_path} is a valid PyTorch checkpoint.")
        model.eval()
else:
    model_ready = False
    print("Model weights s. remove_bg endpoint will not work.")
    print("Provide MODEL_PATH or MODEL_URL, or place resnet101_unet.pth next to this file.")
    model.eval()


def get_loaded_model():
    if not model_ready:
        raise RuntimeError(
            "Background removal model is not loaded. Set MODEL_PATH/ MODEL_URL or place resnet101_unet.pth in app/models_unet/."
        )
    return model
