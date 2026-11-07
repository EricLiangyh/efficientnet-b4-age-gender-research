import argparse
from collections import Counter
from .data import create_manifest


def main():
    p = argparse.ArgumentParser(description='Create a stable, exact-duplicate-grouped train/val/test manifest')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', default='data/manifest.csv')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--val-ratio', type=float, default=0.1)
    args = p.parse_args()
    rows = create_manifest(args.data_root, args.output, args.seed, args.train_ratio, args.val_ratio)
    print(dict(Counter(row['split'] for row in rows)))


if __name__ == '__main__':
    main()
