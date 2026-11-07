import csv
import hashlib
import random
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def transform(train=False, augment=True, size=380):
    steps = [T.Resize((size, size))]
    if train and augment:
        steps += [T.RandomHorizontalFlip(), T.RandomRotation(15),
                  T.RandomPerspective(0.2, p=0.5),
                  T.ColorJitter(0.3, 0.3, 0.3, 0.1), T.RandomGrayscale(p=0.1),
                  T.RandomApply([T.GaussianBlur(3)], p=0.2)]
    steps += [T.ToTensor(), T.Normalize(MEAN, STD)]
    if train and augment:
        steps += [T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')]
    return T.Compose(steps)


def validate_rows(rows):
    paths, hashes = set(), {}
    for row in rows:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Manifest paths must remain relative to the data root')
        if row['path'] in paths:
            raise ValueError(f'Duplicate path: {row["path"]}')
        paths.add(row['path'])
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Unknown split')
        if int(row['gender']) not in {0, 1} or not 0 <= float(row['age']) <= 120:
            raise ValueError(f'Invalid label: {row["path"]}')
        key = row['sha256']
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid SHA256')
        current = (row['split'], float(row['age']), int(row['gender']))
        if key in hashes and hashes[key] != current:
            raise ValueError('Same image has inconsistent labels or crosses splits')
        hashes[key] = current


def create_manifest(root, output, seed=42, train_ratio=0.8, val_ratio=0.1):
    root = Path(root).resolve()
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError('Ratios must leave non-empty train/val/test fractions')
    groups = {}
    for source in ['UTKFace', 'AFAD', 'imdb']:
        for path in sorted((root / source).glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            parts = path.name.split('_')
            age, gender = float(parts[0]), int(parts[1])
            with Image.open(path) as image:
                image.verify()
            row = {'path': path.relative_to(root).as_posix(), 'age': age,
                   'gender': gender, 'source': source, 'sha256': file_hash(path)}
            groups.setdefault(row['sha256'], []).append(row)
    if len(groups) < 10:
        raise ValueError('At least 10 unique images are required for a three-way split')
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_train = int(len(keys) * train_ratio)
    n_val = int(len(keys) * val_ratio)
    if min(n_train, n_val, len(keys) - n_train - n_val) < 1:
        raise ValueError('Split has an empty partition; adjust ratios or add data')
    rows = []
    for index, key in enumerate(keys):
        split = 'train' if index < n_train else ('val' if index < n_train + n_val else 'test')
        rows.extend(dict(row, split=split) for row in groups[key])
    rows.sort(key=lambda row: row['path'])
    validate_rows(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'age', 'gender', 'source', 'sha256', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_manifest(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    validate_rows(rows)
    for row in rows:
        row['age'] = float(row['age'])
        row['gender'] = int(row['gender'])
    return rows


class FaceDataset(Dataset):
    def __init__(self, root, manifest, split, training=False, augment=True, size=380, verify_hashes=False):
        self.root = Path(root).resolve()
        self.rows = [r for r in read_manifest(manifest) if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No samples in {split}')
        self.transform = transform(training, augment, size)
        for row in self.rows:
            path = (self.root / row['path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f'Missing or unsafe sample: {row["path"]}')
            if verify_hashes and file_hash(path) != row['sha256']:
                raise ValueError(f'Image changed: {row["path"]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            return self.transform(image), float(row['age']), int(row['gender'])
