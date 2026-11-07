# legacy：历史实验脚本索引

[返回实验导航](../README.md) · [查看日志](../logs/README.md)

这里的文件按实验用途改名保存，保留历史算法行为；部分注释做过表述整理。因此，如果要看本次提供的两个**逐字节保留原件**，请看根目录的 [preprocess_pro.py](../../preprocess_pro.py) 与 [train_pro.py](../../train_pro.py)。

| 脚本 | 用途 | 对应日志 |
| --- | --- | --- |
| [b4_35.py](b4_35.py) | C3：B4 的 35 轮划分训练与验证；所选权重按来源与此目录实验关联 | [b4_35.csv](../logs/b4_35.csv) |
| [b4_15.py](b4_15.py) | B4，15 轮对比基线 | [b4_15.csv](../logs/b4_15.csv) |
| [b0_15.py](b0_15.py) | B0，15 轮 | [b0_15.csv](../logs/b0_15.csv) |
| [resnet50_15.py](resnet50_15.py) | ResNet-50，15 轮 | [resnet50_15.csv](../logs/resnet50_15.csv) |
| [vgg16_15.py](vgg16_15.py) | VGG-16，15 轮 | [vgg16_15.csv](../logs/vgg16_15.csv) |
| [b4_no_augmentation_15.py](b4_no_augmentation_15.py) | B4，去掉数据增强，15 轮 | [b4_no_augmentation_15.csv](../logs/b4_no_augmentation_15.csv) |
| [b4_gender_weight_1_15.py](b4_gender_weight_1_15.py) | B4，性别损失权重改为 1，15 轮 | [b4_gender_weight_1_15.csv](../logs/b4_gender_weight_1_15.csv) |
| [b4_no_alignment_15.py](b4_no_alignment_15.py) | B4，未对齐数据，15 轮 | [b4_no_alignment_15.csv](../logs/b4_no_alignment_15.csv) |
| [preprocess_no_alignment.py](preprocess_no_alignment.py) | 去除对齐实验的数据准备 | 不直接产生训练日志 |
| [dataset_statistics.py](dataset_statistics.py) | 统计数据分布 | 需原数据集 |
| [preprocess_original.py](preprocess_original.py) | 归档版原始预处理 | 原名完整版本在根目录 |
| [train_full_data_original.py](train_full_data_original.py) | 全数据训练版本，没有验证划分 | 原名完整版本在根目录 |
| [evaluate_original.py](evaluate_original.py) | 历史评估脚本，从同一数据目录抽取子集 | 不能据此认定测试成员与训练隔离 |

这些脚本使用原环境的绝对 Linux 路径、各自数据加载与异常处理方式。新代码中的 JSON 配置不会改变它们。要开展新的可追溯实验，请使用 [统一代码](../../age_gender/README.md)；要核对过去运行，请按本表将脚本与日志对应阅读。
