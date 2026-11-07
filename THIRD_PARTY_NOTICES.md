# Third-party sources and licensing status

No repository-wide license has been chosen. Private hosting does not change third-party terms. Do not assume the trained checkpoint or source datasets are cleared for public or commercial redistribution.

| Source | Role | Reference |
| --- | --- | --- |
| PyTorch / Torchvision | Training framework, EfficientNet implementation and ImageNet initialization | https://pytorch.org/ ; https://github.com/pytorch/vision |
| EfficientNet | Backbone architecture | Tan and Le, ICML 2019: https://proceedings.mlr.press/v97/tan19a.html |
| UTKFace | Age/gender-labelled data source | https://susanqq.github.io/UTKFace/ |
| AFAD | Age/gender-labelled data source | https://github.com/afad-dataset/tarball |
| IMDB-WIKI | Source family for the locally prepared IMDB subset; exact cleaned subset is not distributed | https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ |
| InsightFace | Optional face detection/alignment preprocessing | https://github.com/deepinsight/insightface |
| Pillow, NumPy, tqdm, Matplotlib | Image loading, numerical processing and visualization | Their respective upstream projects and package distributions |

Consult each original source before downloading or using data/models. InsightFace's source code and pretrained-model terms are distinct; pretrained models are described upstream as for non-commercial research. Dataset pages also specify their own usage conditions. This repository distributes neither datasets nor InsightFace pretrained assets.

The selected age/gender checkpoint is a research artifact trained using third-party data and ImageNet initialization. Its presence in this private repository does not grant any additional license. File integrity and source mapping are in `weights/manifest.json` and `docs/provenance.md`.

## Authorized demonstration photograph

`weights/example_face.png` was supplied by the project owner, who explicitly confirmed permission to use the photograph for this repository. It is preserved unchanged and is included as an input-preparation reference. No general redistribution license is granted to visitors by this permission; no endorsement or model-accuracy claim is implied. This photograph is separate from the training datasets.
