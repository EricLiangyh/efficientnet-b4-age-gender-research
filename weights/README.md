# 已训练模型 / Trained model

[返回首页](../README.md) · [模型卡](../docs/model-card.md) · [来源说明](../docs/provenance.md)

| 文件 | 用途 |
| --- | --- |
| [efficientnet_b4_c3_epoch35.pth](efficientnet_b4_c3_epoch35.pth) | 选定的 C3 第 35 轮 B4 参数，约 71 MB |
| [manifest.json](manifest.json) | 文件大小、SHA256，以及另一历史权重的区别 |

权重按目录来源关联 [C3 训练脚本](../experiments/legacy/b4_35.py) 与 [日志](../experiments/logs/b4_35.csv)。根目录全数据 `train_pro.py` 是另一版本，不能直接认定为本权重的生成代码。

在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python predict.py --image path/to/face_crop.jpg --device cpu
```

输入为已裁剪的人脸图片；没有人脸检测环节。输出为年龄估计、0/1 分类标签和未经校准的类别概率。单张预测不衡量准确率。

`predict.py` 仅定义加载这份权重所需的 B4 结构，以 `weights_only=True` 严格加载参数，不额外下载 ImageNet 权重。它不是另一套训练实现。

No project-wide license is selected. Dataset and pretraining terms apply separately. Successful loading and finite outputs do not establish accuracy on unseen data.
