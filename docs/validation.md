# Repository validation

Checked locally on 2026-11-07 (Windows, Python 3.14, CPU).

## Completed

- Five unit checks passed: known metric values; stable split round-trip and duplicate grouping; cross-split duplicate rejection; eight historical CSV logs; strict C3 checkpoint loading and finite forward outputs at 380×380.
- Historical figures and tables regenerated from all eight committed CSV files.
- A separate synthetic 20-image smoke experiment completed one EfficientNet-B0 epoch, validation, checkpoint saving and test evaluation with a matching manifest hash. Images were generated colored squares. Its numerical metrics are not research results and are excluded from the repository.
- Source files checked for excluded document/course terms and local Windows paths.
- Dataset images, account database, private media, frontend and backend excluded.

## Not completed

- Full research training or full-dataset reevaluation of historical checkpoints.
- Reconstructing the original historical sample-level train/validation membership.
- CUDA training, CUDA timing, external test accuracy, identity-level isolation or multi-seed significance analysis.
- End-to-end optional InsightFace preprocessing (requires separately installed dependencies and upstream assets).
- Running the GitHub Actions workflow remotely; workflow configuration is provided.

Successful structural and synthetic checks validate the software path, not the historical performance claim. Environment package versions are recorded in `validation_environment.json`; this is an observed environment, not a portable lockfile.
