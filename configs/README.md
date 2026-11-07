# configs：模型训练参数

[返回首页](../README.md) · [详细调参指南](../docs/tuning.md) · [训练入口](../age_gender/train.py)

这些 JSON 文件告诉训练程序“用什么模型、训练多少轮、怎样优化”。它们本身不是训练代码，也不是历史实验当时自动导出的配置。历史运行以 [归档脚本](../experiments/legacy/README.md) 和 [日志](../experiments/README.md) 为依据。

## 先选实验，再选配置

| 研究阶段 | 配置 | 相对 B4 15 轮基线的变化 |
| --- | --- | --- |
| 模型家族比较 | [b4_15.json](b4_15.json)、[resnet50_15.json](resnet50_15.json)、[vgg16_15.json](vgg16_15.json) | 更换骨干；均为 15 轮 |
| EfficientNet 规模比较 | [b4_15.json](b4_15.json)、[b0_15.json](b0_15.json) | B4 改为 B0；均为 15 轮 |
| 数据增强消融 | [b4_no_augmentation_15.json](b4_no_augmentation_15.json) | `augmentation` 改为 `false` |
| 任务权重比较 | [b4_gender_weight_1_15.json](b4_gender_weight_1_15.json) | `gender_loss_weight` 从 6 改为 1 |
| 完整配置的新训练 | [b4.json](b4.json) | 训练 35 轮 |

人脸对齐是在数据准备阶段执行的，因此没有一个 JSON 开关可以自动完成“去掉对齐”的公平对比。需要准备对应未对齐数据，并保持样本身份及划分一致；历史材料在 [人脸对齐实验](../docs/experiments.md)。

## 参数词典

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `backbone` | `efficientnet_b4` | 共享特征提取网络 |
| `image_size` | 380 | 输入图片的宽和高 |
| `batch_size` | 64 | 每次优化使用的图片数量 |
| `epochs` | 35 | 完整遍历训练数据的次数 |
| `lr_max` | 0.0004 | OneCycleLR 的最高学习率 |
| `weight_decay` | 0.05 | AdamW 的权重衰减系数 |
| `gender_loss_weight` | 6.0 | 性别分类损失在联合目标中的权重 |
| `label_smoothing` | 0.1 | 分类损失的标签平滑系数 |
| `dropout` | 0.3 | EfficientNet/ResNet 特征端的 dropout；新 VGG 实现保留其原分类器中的 dropout，当前不读取此值来改写 VGG dropout |
| `gradient_clip` | 1.0 | 梯度范数裁剪上限 |
| `augmentation` | true | 是否启用新代码中的复合训练增强 |
| `pretrained` | true | 是否用 Torchvision 默认预训练权重初始化 |
| `seed` | 42 | 新训练中 Python、NumPy、PyTorch 的随机种子；数据划分种子由 `split.py --seed` 单独指定 |

## 这些值怎样生效

```powershell
python -m age_gender.train --config configs/b4_15.json --data-root data/aligned --manifest data/manifest.csv --output runs/b4_15 --verify-hashes
```

`--config` 选择 JSON；`--epochs` 命令参数可以覆盖 JSON 中的轮数。`--workers`、`--device`、`--output` 是训练命令参数，不放在这些 JSON 中。每次运行的实际配置会写入输出目录的 `run.json` 和权重文件。

默认骨干比较保持 380×380 输入；并非自动采用每种骨干的原始推荐分辨率。新 VGG/ResNet 参数化也不保证等同历史脚本。详见 [复现协议](../docs/reproducibility.md)。
