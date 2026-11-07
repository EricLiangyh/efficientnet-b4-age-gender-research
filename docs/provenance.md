# Experiment and checkpoint provenance

## Selected checkpoint

`weights/efficientnet_b4_c3_epoch35.pth` was copied from the C3 35-epoch experiment folder and is paired by folder provenance with `experiments/logs/b4_35.csv` and `experiments/legacy/b4_35.py`. The SHA256 and byte size are recorded in `weights/manifest.json`. A successful strict state-dictionary load establishes architectural compatibility, not training provenance or prediction accuracy.

Another historical checkpoint used the same `pro_eb4_epoch_35.pth` basename but has a different SHA256. Its associated supplied training script concatenates all data without holding out samples. Its hash is preserved as `historical_full_data` in the manifest; its weights are not included. The selected model is not assumed interchangeable with it.

## Historical metrics

Eight CSV files record validation metrics from separate runs. C3 uses `random_split` with seed 42 and an 80/20 split. However, the file list was built with unsorted `os.listdir` and no persisted membership manifest was found. Same seed alone cannot establish the same split after directory copies or across training/evaluation scripts.

The historical evaluation script selects 20% from the full data directory. Applied to a full-data-trained checkpoint, this includes seen images. Applied to C3, membership still requires the original ordering to be verified. Therefore its results are not promoted as verified independent-test evidence.

The 20% partition was used during training to monitor validation metrics. It should be described as validation, not a final untouched test set. Identity overlap, exact/near duplicates and identity-level leakage were not resolved by image-level random splitting.

## Verbatim original source files

Root-level `preprocess_pro.py` and `train_pro.py` preserve the two supplied files byte-for-byte, including comments, configuration and line endings. `.gitattributes` disables text conversion specifically for these files. SHA256 and sizes are in `original_scripts.json`. No wording substitutions were made to these root-level originals. They are separate from the renamed archive and from the unified new package.

## Archived versus new implementation

Archived scripts preserve the original experimental behavior, including absolute Linux paths, recursive bad-image replacement, unsorted enumeration and the original geometric crop logic. Wording-only substitutions removed document-specific presentation terminology; filenames were changed to describe each experiment. The archive is evidence, not the recommended executable API.

The new package is a maintainable implementation of the research approach. It saves sorted, hash-grouped sample manifests; fails explicitly on invalid images/labels; separates three splits; stores configuration and manifest hashes; loads weights without redundant downloads; and uses safe weights-only deserialization for attribute checkpoints. CPU execution disables CUDA AMP. These changes mean new results must be reported separately from historical logs.

The new alignment utility uses the detector's five-point eye landmarks and transforms the bounding-box corners with the image before cropping. The old utility uses 106-point indices and crops the rotated image using the original box. New preprocessing output will differ. Do not attach historical performance to this revised preprocessing without running an experiment.

No original dataset or original split-membership manifest is included. No full-dataset reevaluation or full research retraining was performed while preparing this repository.
