import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps
import torch
from .data import transform
from .model import load_model


def main():
    p = argparse.ArgumentParser(description='Estimate attributes from an already-cropped face image')
    p.add_argument('--image', required=True)
    p.add_argument('--checkpoint', default='weights/efficientnet_b4_c3_epoch35.pth')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output')
    args = p.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    size = payload.get('config', {}).get('image_size', 380)
    model = load_model(args.checkpoint, args.device)
    with Image.open(args.image) as image:
        x = transform(size=size)(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        age, logits = model(x)
    probs = logits.softmax(1)[0].cpu().tolist()
    result = {'estimated_age_years': age.item(), 'gender_label': int(logits.argmax(1).item()), 'label_probabilities': probs, 'note': 'Dataset-label estimate; probabilities are not calibrated confidence.'}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
