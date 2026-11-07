# Data preparation

Obtain UTKFace, AFAD and the intended IMDB-derived subset from authorized sources. Dataset images are not included. Follow original usage terms in [third-party sources](../THIRD_PARTY_NOTICES.md). The exact historical cleaned IMDB subset and its filtering provenance have not been reconstructed.

## Label conventions

The aligned loader expects filenames beginning with `age_gender_`, where age is numeric and gender is 0 or 1. UTKFace uses its own age/gender filename fields. Historical AFAD preparation mapped folder 111 to 0 and 112 to 1. The historical IMDB preparation used a local CSV with `filename`, `age`, and `gender`; audit its label encoding before reuse. Reject unknown/missing labels rather than silently mapping them to 0.

## New alignment utility

Install optional preprocessing dependencies with `python -m pip install -e ".[preprocess]"`. InsightFace may require a compiler on some systems; its assets are obtained separately under upstream terms. CPU ONNX Runtime is the default; CUDA preprocessing requires a matching GPU runtime and `--provider CUDAExecutionProvider`.

Create a CSV index with columns `path,age,gender,source`, for example:

```csv
path,age,gender,source
UTKFace/25_0_example.jpg,25,0,UTKFace
AFAD-Full/28/112/example.jpg,28,1,AFAD
```

```powershell
python -m age_gender.preprocess --index data/raw_index.csv --raw-root data/raw --output-root data/aligned
python -m age_gender.split --data-root data/aligned --output data/manifest.csv
```

The utility requires exactly one detected face, applies eye-based rotation and transforms the bounding box before a 35% expanded crop, then resizes to 380×380. It records failures in `preprocessing_report.csv`. No synthetic fallback face or silent random sample substitution is used. This differs from historical preprocessing; evaluate it as a new experiment.

## Split protocol

The manifest records relative path, age, gender, source, SHA256 and split. Filenames are sorted, exact-image hashes are grouped, and groups are shuffled using seed 42. Default group allocation is 80/10/10, so image percentages may vary when duplicates exist. At least ten unique images are required. Conflicting labels for identical files are rejected.

This is image-hash grouping, not identity-aware grouping. Near duplicates, differently encoded copies and different photographs of one person may cross partitions. For identity-independent experiments, derive an audited grouped manifest from identity metadata, then run the same validators. Do not publish identity metadata or face images by default.
