"""Load the archived B4 checkpoint and predict an already-cropped face.

Usage: python weights/predict.py --image path/to/face_crop.jpg
Training and preprocessing remain in the two original scripts.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms


class AgeGenderModel(nn.Module):
    """Exact B4 head structure required by the archived C3 state dictionary."""
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.3, inplace=True), nn.Identity())
        self.age_head = nn.Linear(features, 1)
        self.gender_head = nn.Linear(features, 2)

    def forward(self, images):
        features = self.backbone(images)
        return self.age_head(features), self.gender_head(features)


def main():
    parser = argparse.ArgumentParser(description='Age/gender prediction for a cropped face; no face detection')
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', default=str(Path(__file__).resolve().parent / 'efficientnet_b4_c3_epoch35.pth'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    model = AgeGenderModel()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(args.device).eval()
    transform = transforms.Compose([
        transforms.Resize((380, 380)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(args.image) as image:
        batch = transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        age, logits = model(batch)
    print(json.dumps({
        'estimated_age_years': age.item(),
        'gender_label': int(logits.argmax(1).item()),
        'label_probabilities': logits.softmax(1)[0].cpu().tolist(),
        'note': 'Dataset labels 0=male, 1=female; probabilities are uncalibrated.',
    }, indent=2))


if __name__ == '__main__':
    main()
