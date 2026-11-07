# 模型权重与使用 / Selected model weights

## 先看这里：这个目录里是什么

| 文件 | 用途 |
| --- | --- |
| [efficientnet_b4_c3_epoch35.pth](efficientnet_b4_c3_epoch35.pth) | 已训练的 C3 B4 模型参数，约 71 MB；它是二进制权重，不是 Python 源代码 |
| [manifest.json](manifest.json) | 选定权重的文件大小、SHA256，以及另一历史权重的区别记录 |

权重包含模型参数，使用时仍需要 [模型结构代码](../age_gender/model.py)。它按来源关联 [C3 训练脚本](../experiments/legacy/b4_35.py) 与 [C3 日志](../experiments/logs/b4_35.csv)，不能直接当成根目录全数据 `train_pro.py` 的输出。

### 用现有模型预测一张图片

安装项目依赖后，从仓库根目录运行：

```powershell
python -m age_gender.predict --image path/to/face_crop.jpg --checkpoint weights/efficientnet_b4_c3_epoch35.pth --device cpu
```

输入必须是已裁剪的人脸图片，不是让该命令自动在多人照片中检测人脸。输出包含年龄估计、二元数据集标签与未经校准的类别概率。单张预测不需要真实标签，也不会衡量准确率。

若要对一组有真实标签的图片评估 MAE、准确率等，请先看 [评估指南](../docs/evaluation.md)。历史权重缺少原始划分清单，需要按指南明确处理其来源限制。

[返回首页](../README.md) · [第一次阅读路线](../docs/reading-guide.md) · [代码导航](../age_gender/README.md)

`efficientnet_b4_c3_epoch35.pth` is the 35-epoch C3 EfficientNet-B4 state dictionary. It is versioned in this private repository and loads with `weights_only=True` and strict keys. Byte size and SHA256 are in [manifest.json](manifest.json).

The manifest also identifies a different historical full-data checkpoint that is deliberately not included. The chosen checkpoint is paired with its folder's training code and CSV log; those artifacts do not establish independent test accuracy. See [model card](../docs/model-card.md) and [provenance](../docs/provenance.md).

No weight license has been selected. Original dataset and pretraining terms apply separately. No detector or web application is needed for the cropped-face prediction command.
