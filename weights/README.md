# Selected model weights

`efficientnet_b4_c3_epoch35.pth` is the 35-epoch C3 EfficientNet-B4 state dictionary. It is versioned in this private repository and loads with `weights_only=True` and strict keys. Byte size and SHA256 are in [manifest.json](manifest.json).

The manifest also identifies a different historical full-data checkpoint that is deliberately not included. The chosen checkpoint is paired with its folder's training code and CSV log; those artifacts do not establish independent test accuracy. See [model card](../docs/model-card.md) and [provenance](../docs/provenance.md).

No weight license has been selected. Original dataset and pretraining terms apply separately. No detector or web application is needed for the cropped-face prediction command.
