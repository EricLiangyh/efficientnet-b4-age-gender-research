# EfficientNet-B4 Age and Gender Estimation Research

Multi-task facial age regression and gender-label classification with a shared EfficientNet-B4 backbone.

[中文](README.zh-CN.md)

## Start with the research implementation

| Step | File | Purpose |
| --- | --- | --- |
| **1. Prepare data** | [preprocessing/](preprocessing/README.md) | Complete original preprocessing: dataset scanning, face alignment and cropping |
| **2. Train the model** | [training/](training/README.md) | Complete original EfficientNet-B4 dual-head training implementation |
| **3. Use trained weights** | [weights/](weights/README.md) | Selected C3 epoch-35 checkpoint, source and usage |

The two original source files are unchanged. The selected checkpoint comes from the separate [C3 split-based training script](experiments/legacy/b4_35.py), not the original full-data training run. [Source and integrity records](docs/provenance.md) explain the distinction.

## Method

The model shares 1792-dimensional EfficientNet-B4 features between a linear age head and a two-logit classification head. Historical training uses `L1(age) + 6 × CrossEntropy(gender)`, label smoothing 0.1, AdamW, OneCycleLR and composite augmentation.

[Data preparation, training parameters and evaluation notes](docs/methodology.md)

## Experiments

1. **Backbone selection, 15 epochs:** B4 vs ResNet-50 vs VGG-16.
2. **EfficientNet scaling, 15 epochs:** B4 vs B0.
3. **Training strategies, 15 epochs each:** augmentation, task-loss weighting and alignment.
4. **Final B4 training, 35 epochs:** loss and validation metrics shown separately.

[Experiment code and logs](experiments/README.md) · [Grouped results and figures](docs/experiments.md)

The final historical validation log records **MAE 3.4074 years** and **accuracy 98.2064%**. These are archived validation observations, not freshly verified independent-test results. The original sample-level partition manifest is unavailable.

![Final B4 training](experiments/figures/final_training_35.png)

## Try the checkpoint

With compatible PyTorch/Torchvision installed, install the small inference dependency set and supply an already-cropped face:

```powershell
python -m pip install -r requirements.txt
python predict.py --image path/to/face_crop.jpg
```

`predict.py` is a single-file inference helper; it does not detect faces or measure accuracy. Dataset labels are 0=male and 1=female. Probabilities are uncalibrated and are not claims about personal gender identity.

## Repository structure

```text
preprocessing/          Original preprocessing code and setup guide
training/               Original training code and setup guide
predict.py               Minimal checkpoint inference
weights/                 Selected trained model and checksums
experiments/             Historical scripts, eight logs and six figures
docs/                    Method, model card, results and provenance
```

No dataset images, private media, web application or generic training framework is included. No project-wide license is selected; [third-party terms](THIRD_PARTY_NOTICES.md) apply separately. See the [model card](docs/model-card.md) for limitations.
