# 模型训练 / Model training

[返回项目首页](../README.md) · [查看完整代码：train_pro.py](train_pro.py) · [上一步：数据预处理](../preprocessing/README.md) · [已训练模型](../weights/README.md)

本目录只放原始训练脚本与这份说明。脚本的内容、参数、注释和换行格式均保持不变，校验记录见 [original_scripts.json](../docs/original_scripts.json)。

## 1. 训练流程

```text
预处理导出的 UTKFace / AFAD / imdb 图片
    ↓ 从文件名读取年龄和分类标签
合并三个数据源 + 训练数据增强
    ↓
EfficientNet-B4 共享骨干
    ↓ 1792 维共享特征 + Dropout
    ├── 年龄回归头：1 个输出
    └── 性别分类头：2 个 logits
    ↓ 联合损失、反向传播和优化
每轮显示训练指标并保存模型参数
```

这是**全数据训练版本**，没有划分训练/验证/测试集，也没有验证循环。仓库当前选定的权重来自另一组 [C3 划分训练脚本](../experiments/legacy/b4_35.py) 和 [对应日志](../experiments/logs/b4_35.csv)，不能直接认定为此原始脚本生成的同一份模型。

## 2. 数据目录与标签

原配置：

```python
BASE_DIR = "/root/my_age_project"
DATA_DIR = f"{BASE_DIR}/data_aligned"
SAVE_DIR = f"{BASE_DIR}/checkpoints_pro_v1"
```

`DATA_DIR` 对应 [预处理脚本](../preprocessing/preprocess_pro.py) 的 `TARGET_DIR`：

```text
data_aligned/
├── UTKFace/
├── AFAD/
└── imdb/
```

脚本扫描各子目录中的 `.jpg`、`.png`、`.jpeg`，文件名以 `年龄_性别_` 开头。例如 `28_1_afad_example.jpg` 解析为年龄 28、分类标签 1。标签约定为 0=male、1=female。图片使用 Pillow 转为 RGB，再执行变换。

三个存在的数据目录通过 `ConcatDataset` 合并，每批随机打乱，不做数据源平衡采样。损坏图片或标签解析异常会递归随机换一个样本，而不是生成错误清单；运行前应检查文件和标签质量。

## 3. 模型结构

| 部分 | 原始实现 |
| --- | --- |
| 骨干 | Torchvision `efficientnet_b4` |
| 初始化 | `EfficientNet_B4_Weights.IMAGENET1K_V1` |
| 输入 | RGB，380×380，ImageNet 均值/标准差归一化 |
| 共享特征 | 1792 维 |
| 特征端正则化 | Dropout，概率 0.3 |
| 年龄头 | `Linear(1792, 1)`，连续回归，无输出范围限制 |
| 分类头 | `Linear(1792, 2)`，输出两个 logits |

训练时分类头不需要手动加 Softmax，因为 `CrossEntropyLoss` 接收 logits。年龄头可能输出负数或超出合理范围的值；原代码没有截断处理。

## 4. 训练参数在哪里配置

在工作副本的脚本顶部修改路径与这些常量，无独立 JSON 配置或命令行训练参数：

| 参数 | 原始值 | 作用 |
| --- | --- | --- |
| `BATCH_SIZE` | 64 | 每次优化使用的图片数量 |
| `EPOCHS` | 35 | 完整训练轮数 |
| `LR_MAX` | 4e-4 | OneCycleLR 最高学习率 |
| `GEN_LOSS_WEIGHT` | 6.0 | 性别分类损失的权重 |
| `DEVICE` | 自动选择 CUDA/CPU | 模型和数据放置设备；AMP 仍保留原 CUDA 设置 |
| `SAVE_DIR` | `checkpoints_pro_v1` | 每轮权重保存位置 |

其他设置在 `train()` 内：AdamW 的 `weight_decay=0.05`，OneCycleLR 的 `pct_start=0.2`，标签平滑 0.1，梯度裁剪上限 1.0。DataLoader 使用 `num_workers=16`、`pin_memory=True`、`prefetch_factor=3`。

更换路径、批量大小或数据加载进程数量时，请记录实际配置并使用新的输出目录。若把 `num_workers` 改为 0，应同时处理 `prefetch_factor` 设置，不能直接保留多进程预取参数。

## 5. 数据增强

`get_train_transform()` 按以下顺序处理：

| 步骤 | 设置 |
| --- | --- |
| Resize | 380×380 |
| 水平翻转 | 概率 0.5 |
| 随机旋转 | ±15° |
| 透视变化 | 强度 0.2，概率 0.5 |
| 颜色扰动 | 亮度/对比度/饱和度 0.3，色调 0.1 |
| 灰度 | 概率 0.1 |
| 高斯模糊 | 核大小 3，应用概率 0.2 |
| 张量化与归一化 | mean `[0.485,0.456,0.406]`，std `[0.229,0.224,0.225]` |
| 随机擦除 | 概率 0.3，面积比例 0.02～0.2，长宽比 0.3～3.3 |

这些增强在每次训练读取时随机应用，不是再次改写预处理输出图片。

## 6. 损失函数与优化步骤

```text
联合损失 = 年龄 L1 损失 + 6 × 分类交叉熵
```

年龄使用 `L1Loss`；分类使用 `CrossEntropyLoss(label_smoothing=0.1)`。每批依次进行混合精度前向计算、缩放反向传播、解除梯度缩放、梯度裁剪、优化器更新和学习率调度。

原脚本显式使用 `GradScaler('cuda')` 和 `autocast('cuda')`。即使 `DEVICE` 自动回退为 CPU，也不代表完整脚本已经针对 CPU 调整；原实现面向 CUDA 训练环境。

## 7. 如何运行

需要兼容的 PyTorch、Torchvision、Pillow、tqdm 和对应 GPU 环境。`torch.amp` 接口需由实际安装版本支持。根目录 `requirements.txt` 是预测入口的最小依赖，不包含 tqdm；预处理另有依赖，见 [预处理说明](../preprocessing/README.md)。

原训练初始化会获取 ImageNet 权重，需能够下载或提前缓存。确认预处理数据、路径、输出位置和环境后，从仓库根目录运行：

```bash
python training/train_pro.py
```

## 8. 控制台与输出文件怎样理解

控制台 `mae` 是本轮截至当前批次的**训练年龄平均绝对误差**，`acc` 是对应训练分类准确率；两者在随机增强、训练模式下统计，并非验证或独立测试成绩。

```text
checkpoints_pro_v1/
├── pro_eb4_epoch_1.pth
├── pro_eb4_epoch_2.pth
└── ... pro_eb4_epoch_35.pth
```

每轮保存 `model.state_dict()`，只有模型参数；不保存优化器、学习率调度器、原始划分清单或 CSV 训练日志。原脚本不自动选择最佳轮、不自动恢复中断训练。重复运行相同输出目录会覆盖同名权重，因此新实验应使用新目录。

## 9. 参数比较与研究结果

本脚本未固定随机种子，也未隔离验证集。若用于新研究，应先设计并记录训练/验证/测试成员，再在验证集上比较参数，最终才评估独立测试集。修改原脚本后的实验不能仅凭文件名认定为已复现历史结果。

已有的骨干比较、增强/任务权重/对齐消融和最终 C3 实验，请按 [实验索引](../experiments/README.md) 查看各自脚本、日志与图表；来源限制见 [provenance.md](../docs/provenance.md)。

English: This folder preserves the original full-data training script unchanged. Configure paths and constants inside a working copy, prepare aligned datasets and a compatible CUDA environment, then run `python training/train_pro.py`. This script has no held-out validation loop and is distinct from the archived C3 run associated with the distributed checkpoint.
