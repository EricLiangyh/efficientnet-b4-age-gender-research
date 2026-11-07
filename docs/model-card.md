# Model card: EfficientNet-B4 C3 epoch 35

| Field | Description |
| --- | --- |
| Task | Facial age regression and dataset gender-label classification |
| Architecture | Torchvision EfficientNet-B4, shared 1792-dimensional features, dropout 0.3, independent linear heads |
| Input | RGB face crop, resized to 380×380; ImageNet mean/std normalization |
| Output | One unbounded age value in years; two raw classification logits |
| Label convention | 0=male, 1=female, as encoded by the datasets |
| Initialization | ImageNet pretrained weights during historical training |
| Historical optimization | AdamW, lr max 4e-4, weight decay 0.05; OneCycleLR; 35 epochs; batch 64 |
| Objective | L1 age loss + 6 × cross entropy with smoothing 0.1 |
| Checkpoint | `weights/efficientnet_b4_c3_epoch35.pth` |
| Integrity | `weights/manifest.json` |
| Training data sources | UTKFace, AFAD and a locally prepared IMDB-derived subset |
| Measured evidence | Historical validation CSV; no new independent test accuracy verified |
| License | No project-level license selected; source-data/pretraining terms require separate review |

## Intended use

Research experiments on authorized face imagery, educational inspection of multi-task learning, and offline model comparisons with explicitly documented evaluation protocols. This repository does not verify suitability for deployment in any business setting.

## Caveats

Age labels may be noisy and do not necessarily describe perceived age. Predictions may be negative or exceed plausible ages: the head has no output bound. Class probabilities are uncalibrated. Binary dataset labels exclude identities outside those annotations and must not be interpreted as an individual's gender identity. Real-world subgroup performance has not been established.

Historical validation is not an external or identity-independent test. Domain shifts in cameras, pose, lighting, crop size, alignment and age distribution may affect performance. Report failure cases and subgroup sample sizes rather than relying on selected demonstrations.
