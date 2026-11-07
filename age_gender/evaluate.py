import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import FaceDataset, file_hash
from .model import load_model
from .metrics import calculate


def main():
    p = argparse.ArgumentParser(description='Evaluate on an explicit manifest; never regenerate a split')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output', default='runs/evaluation.json')
    p.add_argument('--allow-unverified-manifest', action='store_true', help='Required for legacy checkpoints; does not establish independent performance')
    args = p.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    matched = payload.get('manifest_sha256') == file_hash(args.manifest)
    if not matched and not args.allow_unverified_manifest:
        p.error('Manifest provenance is unverified. Use the original run manifest or explicitly acknowledge a legacy checkpoint.')
    size = payload.get('config', {}).get('image_size', 380)
    data = FaceDataset(args.data_root, args.manifest, args.split, size=size, verify_hashes=True)
    model = load_model(args.checkpoint, args.device)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    true_a, pred_a, true_g, pred_g = [], [], [], []
    with torch.inference_mode():
        for images, ages, genders in loader:
            pa, pg = model(images.to(args.device))
            true_a.extend(ages.tolist())
            true_g.extend(genders.tolist())
            pred_a.extend(pa.squeeze(-1).cpu().tolist())
            pred_g.extend(pg.argmax(1).cpu().tolist())
    result = calculate(true_a, pred_a, true_g, pred_g)
    result.update({'split': args.split, 'checkpoint_sha256': file_hash(args.checkpoint), 'manifest_sha256': file_hash(args.manifest), 'manifest_matches_checkpoint': matched, 'independence_note': 'Split membership matches recorded training manifest; identity overlap still requires audit.' if matched else 'Legacy checkpoint: training membership unknown; this is not verified independent test performance.'})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    with output.with_suffix('.predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'age_true', 'age_pred', 'gender_true', 'gender_pred'])
        writer.writerows((r['path'], a, b, c, d) for r, a, b, c, d in zip(data.rows, true_a, pred_a, true_g, pred_g))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
