import argparse
import json
import platform
import time
import statistics
from pathlib import Path
import torch
from .model import load_model


def main():
    p = argparse.ArgumentParser(description='Attribute-model-only latency with warmup and CUDA synchronization')
    p.add_argument('--checkpoint', default='weights/efficientnet_b4_c3_epoch35.pth')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--output', default='runs/benchmark.json')
    args = p.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.batch_size < 1:
        p.error('Invalid iteration or batch count')
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    x = torch.randn(args.batch_size, 3, 380, 380, device=device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    durations = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        sync()
        for _ in range(args.iterations):
            sync()
            start = time.perf_counter()
            model(x)
            sync()
            durations.append((time.perf_counter() - start) * 1000)
    result = {'scope': 'attribute_model_only_synthetic_tensor', 'device': str(device),
              'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
              'torch': str(torch.__version__), 'batch_size': args.batch_size,
              'warmup': args.warmup, 'iterations': args.iterations,
              'median_batch_ms': statistics.median(durations), 'mean_batch_ms': statistics.mean(durations),
              'note': 'Excludes detection, alignment, decoding and video processing.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
