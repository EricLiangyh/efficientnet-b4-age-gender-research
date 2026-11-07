from pathlib import Path
import torch
from torch import nn
from torchvision import models


class AgeGenderModel(nn.Module):
    """Shared backbone with age regression and two dataset-label logits.

    EfficientNet-B4 state_dict keys match the archived C3 checkpoint.
    Other backbones define new experiments, not reconstructed historical runs.
    """

    def __init__(self, backbone='efficientnet_b4', pretrained=False, dropout=0.3):
        super().__init__()
        if backbone not in {'efficientnet_b4', 'efficientnet_b0', 'resnet50', 'vgg16'}:
            raise ValueError(f'Unsupported backbone: {backbone}')
        self.backbone_name = backbone
        self.backbone = getattr(models, backbone)(weights='DEFAULT' if pretrained else None)
        if backbone.startswith('efficientnet'):
            n = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(nn.Dropout(dropout, inplace=True), nn.Identity())
        elif backbone == 'resnet50':
            n = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Identity())
        else:
            n = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Identity()
        self.age_head = nn.Linear(n, 1)
        self.gender_head = nn.Linear(n, 2)

    def forward(self, x):
        features = self.backbone(x)
        return self.age_head(features), self.gender_head(features)


def load_model(checkpoint, device='cpu', backbone='efficientnet_b4'):
    payload = torch.load(Path(checkpoint), map_location='cpu', weights_only=True)
    config = payload.get('config', {}) if 'model' in payload else {}
    model = AgeGenderModel(config.get('backbone', backbone), dropout=config.get('dropout', 0.3))
    model.load_state_dict(payload['model'] if 'model' in payload else payload, strict=True)
    return model.to(device).eval()
