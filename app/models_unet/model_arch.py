import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import urllib.request

def ensure_model_weights():
    """
    Ensures the model weights are available. If they are missing, downloads them from GitHub Releases.
    
    Returns:
        str: Path to the file with the model weights
    """
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(model_dir, "resnet101_unet.pth")
    
    github_url = "https://github.com/Artem817/Fastapi-image-service/releases/download/v1.0.0-rc1/resnet101_unet.pth"
    
    if not os.path.exists(weights_path):
        print("Model weights not found. Attempting to download from GitHub Releases...")
        try:
            print(f"   Downloading from: {github_url}")
            urllib.request.urlretrieve(github_url, weights_path)
            print(f"✓ Download complete! Saved to: {weights_path}")
            return weights_path
        except urllib.error.URLError as e:
            print(f"✗ Failed to download weights from GitHub: {e}")
            print(f"   Please manually download and place the file at: {weights_path}")
            print(f"   Or create a Release on GitHub with the .pth file")
            return None
        except Exception as e:
            print(f"✗ Unexpected error during download: {e}")
            return None
    else:
        print(f"✓ Model weights found at: {weights_path}")
        return weights_path

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

model = ResNet101Unifier(n_classes=1).to(device)

weights_path = ensure_model_weights()

if weights_path and os.path.exists(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print(f"✓ Model loaded successfully and set to eval mode")
    except Exception as e:
        print(f"✗ Error loading model weights: {e}")
        print(f"  Make sure {weights_path} is a valid PyTorch checkpoint.")
        model.eval()
else:
    print(f"⚠ Model weights unavailable. remove_bg endpoint will not work.")
    print(f"  To fix this:")
    print(f"  1. Download or train resnet101_unet.pth")
    print(f"  2. Create a GitHub Release and upload the file")
    print(f"  3. Update the github_url in ensure_model_weights()")
    print(f"  4. Restart the application")
    model.eval()

