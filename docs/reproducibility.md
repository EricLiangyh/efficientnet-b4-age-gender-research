# Reproducibility protocol

## Two distinct evidence tracks

1. `experiments/legacy` and `experiments/logs` preserve historical experiments. Their missing split membership prevents exact historical replication from this repository alone.
2. `age_gender` and `configs` provide a new pipeline that records manifests and configuration. These changes are not claimed to reproduce old numerical results without retraining.

## New run procedure

From the repository root, install `python -m pip install -e .` after choosing your PyTorch CPU/CUDA environment. Core commands also work directly from the root without an editable install if dependencies already exist.

```powershell
python -m age_gender.split --data-root data/aligned --output data/manifest.csv
python -m age_gender.train --config configs/b4.json --data-root data/aligned --manifest data/manifest.csv --output runs/b4 --workers 0 --verify-hashes
python -m age_gender.evaluate --checkpoint runs/b4/best.pth --data-root data/aligned --manifest runs/b4/manifest.csv --split test --output runs/b4/test.json
python -m age_gender.plot_results
```

The best checkpoint is selected by validation age MAE. Do not tune on the test partition. Training writes `training.csv`, `run.json`, a copy of the manifest, `best.pth` and `last.pth`. The checkpoint includes configuration and manifest hash. Evaluation verifies every selected image hash, reports MAE/RMSE, accuracy, per-class precision/recall/F1 and a confusion matrix, and writes per-image predictions alongside the JSON report. Preserve these artifacts privately if sample filenames could reveal personal information.

Use unique output directories; existing training logs are not overwritten. New configs include 15-epoch backbone comparisons and two loss/augmentation ablations. Alignment ablation requires a separately generated unaligned dataset with an independently audited, consistent split; no automated paired alignment experiment is claimed.

The new VGG/ResNet comparison implementations are explicitly new experiments and are not guaranteed to match historical parameterizations. CPU training disables CUDA AMP; GPU AMP uses gradient clipping and skips learning-rate scheduler advancement after overflow-skipped updates. Checkpoints do not include optimizer state, so automatic resumption is not supported.

## Historical checkpoint evaluation

The archived checkpoint has no saved sample manifest. Evaluation therefore rejects it by default. `--allow-unverified-manifest` permits exploratory measurement while marking membership provenance as unverified. This flag cannot convert a new split into an independent test of an old model.

## Timing

```powershell
python -m age_gender.benchmark --device cpu --iterations 100 --output runs/cpu_latency.json
```

Benchmarks use warmup and explicit CUDA synchronization. They measure only attribute-model execution on synthetic tensors; detection, alignment, decoding and video pipelines are excluded. Do not infer end-to-end FPS.

## Checks

Run `python -m unittest discover -s tests -v`. The checkpoint forward-pass test uses a generated tensor and does not measure model accuracy. Reproducible GPU training may still differ by software versions and nondeterministic kernels; record environment and use multiple seeds for comparative claims.
