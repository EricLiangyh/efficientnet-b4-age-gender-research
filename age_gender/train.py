import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import AgeGenderModel
from .data import FaceDataset, file_hash


def epoch(model, loader, device, config, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = [0., 0., 0., 0]
    loss_age = nn.L1Loss()
    loss_gender = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    with torch.set_grad_enabled(training):
        for images, ages, genders in tqdm(loader, leave=False):
            images, ages, genders = images.to(device), ages.to(device).float(), genders.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
                pa, pg = model(images)
                pa = pa.squeeze(-1)
                loss = loss_age(pa, ages) + config['gender_loss_weight'] * loss_gender(pg, genders)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
            n = len(images)
            totals[0] += loss.item() * n
            totals[1] += (pa - ages).abs().sum().item()
            totals[2] += (pg.argmax(1) == genders).sum().item()
            totals[3] += n
    return [v / totals[3] for v in totals[:3]]


def main():
    p = argparse.ArgumentParser(description='Train multi-task age/gender model with an explicit split manifest')
    p.add_argument('--config', default='configs/b4.json')
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', default='runs/b4')
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--verify-hashes', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if config['epochs'] < 1:
        p.error('epochs must be positive')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training.csv').exists():
        raise FileExistsError('Choose a new output directory; existing run will not be overwritten')
    loaders = []
    for split in ['train', 'val']:
        data = FaceDataset(args.data_root, args.manifest, split, split == 'train', config['augmentation'], config['image_size'], args.verify_hashes)
        loaders.append(DataLoader(data, batch_size=config['batch_size'], shuffle=split == 'train', num_workers=args.workers, pin_memory=device.type == 'cuda'))
    model = AgeGenderModel(config['backbone'], config['pretrained'], config['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr_max'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr_max'], total_steps=len(loaders[0]) * config['epochs'], pct_start=0.2)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    metadata = {'config': config, 'manifest_sha256': file_hash(args.manifest), 'torch': str(torch.__version__), 'device': str(device), 'split_counts': dict(zip(['train', 'val'], [len(l.dataset) for l in loaders])), 'status': 'new_run_not_historical_reproduction'}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'manifest.csv').write_bytes(Path(args.manifest).read_bytes())
    best = float('inf')
    with (output / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_loss', 'train_mae', 'train_accuracy', 'val_loss', 'val_mae', 'val_accuracy'])
        for index in range(config['epochs']):
            tr = epoch(model, loaders[0], device, config, optimizer, scheduler, scaler)
            va = epoch(model, loaders[1], device, config)
            writer.writerow([index + 1, *tr, *va])
            stream.flush()
            payload = {'model': model.state_dict(), 'config': config, 'epoch': index + 1, 'manifest_sha256': metadata['manifest_sha256']}
            torch.save(payload, output / 'last.pth')
            if va[1] < best:
                best = va[1]
                torch.save(payload, output / 'best.pth')
            print(f'Epoch {index+1}: val MAE={va[1]:.4f}, accuracy={va[2]:.4%}')


if __name__ == '__main__':
    main()
