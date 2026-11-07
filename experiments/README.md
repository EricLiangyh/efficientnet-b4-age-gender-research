# 实验索引

[返回首页](../README.md) · [分组图表与结果](../docs/experiments.md) · [参数解释](../docs/methodology.md)

这里仅保存支撑研究结果的历史脚本、八组逐轮日志与六张图。`legacy/` 为历史实现，`logs/` 为对应 CSV，`figures/` 为日志生成的图。

| 阶段 | 代码 | 日志 |
| --- | --- | --- |
| B4 基线，15 轮 | [b4_15.py](legacy/b4_15.py) | [CSV](logs/b4_15.csv) |
| ResNet-50，15 轮 | [resnet50_15.py](legacy/resnet50_15.py) | [CSV](logs/resnet50_15.csv) |
| VGG-16，15 轮 | [vgg16_15.py](legacy/vgg16_15.py) | [CSV](logs/vgg16_15.csv) |
| B0，15 轮 | [b0_15.py](legacy/b0_15.py) | [CSV](logs/b0_15.csv) |
| 无增强，15 轮 | [b4_no_augmentation_15.py](legacy/b4_no_augmentation_15.py) | [CSV](logs/b4_no_augmentation_15.csv) |
| 性别损失权重 1，15 轮 | [b4_gender_weight_1_15.py](legacy/b4_gender_weight_1_15.py) | [CSV](logs/b4_gender_weight_1_15.csv) |
| 无对齐，15 轮 | [b4_no_alignment_15.py](legacy/b4_no_alignment_15.py)、[数据处理](legacy/preprocess_no_alignment.py) | [CSV](logs/b4_no_alignment_15.csv) |
| 最终 C3，35 轮 | [b4_35.py](legacy/b4_35.py) | [CSV](logs/b4_35.csv) |

日志中 `train_loss`/`val_loss` 为联合损失，`val_mae` 单位为岁，`val_acc` 为百分比。末轮结果不必然是最佳轮；不同任务权重下的联合损失也不能直接等价比较。

现有图表保留历史 CSV 中的数值，未平滑或修改测量结果。历史脚本保留原环境路径和行为，文件名及部分注释经过整理；根目录两个原件则逐字节保留。
