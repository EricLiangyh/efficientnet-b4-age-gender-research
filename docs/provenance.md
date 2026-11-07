# Experiment and checkpoint provenance

## Selected checkpoint

`weights/efficientnet_b4_c3_epoch35.pth` was copied from the C3 35-epoch experiment folder and is paired by folder provenance with `experiments/logs/b4_35.csv` and `experiments/legacy/b4_35.py`. The SHA256 and byte size are recorded in `weights/manifest.json`. A successful strict state-dictionary load establishes architectural compatibility, not training provenance or prediction accuracy.

Another historical checkpoint used the same `pro_eb4_epoch_35.pth` basename but has a different SHA256. Its associated supplied training script concatenates all data without holding out samples. Its hash is preserved as `historical_full_data` in the manifest; its weights are not included. The selected model is not assumed interchangeable with it.

## Historical metrics

Eight CSV files record validation metrics from separate runs. C3 uses `random_split` with seed 42 and an 80/20 split. However, the file list was built with unsorted `os.listdir` and no persisted membership manifest was found. Same seed alone cannot establish the same split after directory copies or across training/evaluation scripts.

The historical evaluation script selects 20% from the full data directory. Applied to a full-data-trained checkpoint, this includes seen images. Applied to C3, membership still requires the original ordering to be verified. Therefore its results are not promoted as verified independent-test evidence.

The 20% partition was used during training to monitor validation metrics. It should be described as validation, not a final untouched test set. Identity overlap, exact/near duplicates and identity-level leakage were not resolved by image-level random splitting.

## Verbatim original source files

`preprocessing/preprocess_pro.py` and `training/train_pro.py` preserve the two supplied files byte-for-byte, including comments, configuration and line endings. `.gitattributes` disables text conversion specifically for these files. SHA256 and sizes are in `original_scripts.json`. No wording substitutions were made to these original files. They are separate from the renamed historical archive.

## Archived scripts and minimal utilities

Historical comparison scripts preserve original experimental behavior, including Linux paths, recursive bad-image replacement and unsorted enumeration. Their filenames and some comments were adjusted during archiving. The original preprocessing and training scripts remain byte-for-byte unchanged.

The only standalone supporting utility is `weights/predict.py`, which loads the selected B4 checkpoint for cropped-face inference. Existing figures are retained alongside their source CSV logs; no standalone plotting script is included.

No original dataset or split-membership manifest is included. No full-dataset reevaluation or full research retraining was performed while preparing this repository.

## Included interface example

`weights/example_face.png` is a geometric illustration of a fictional face created for this repository, not a photograph or dataset sample. `weights/example_output.json` records its actual CPU inference output, file hashes and environment. There is no ground-truth age or gender; this is an executable interface example and does not establish photographic-domain performance. The generation helper is not part of the research code.
