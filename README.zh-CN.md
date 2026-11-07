# 基于 EfficientNet-B4 的人脸年龄与性别预测模型研究

[English](README.md) · [模型说明](docs/model-card.md) · [实验记录](docs/experiments.md) · [复现协议](docs/reproducibility.md)

本项目研究共享 EfficientNet-B4 骨干下的年龄回归与性别标签分类，围绕多源数据融合、人脸几何预处理、复合数据增强与多任务损失加权开展实验。仓库集中呈现模型、训练方法与实验依据。

## 原始预处理与训练代码

你提供的两个原始脚本已按原名放在仓库根目录，**完整保留，未改动任何内容**：

- [`preprocess_pro.py`](preprocess_pro.py)：扫描 UTKFace、AFAD 与 IMDB 数据，使用 InsightFace 106 点关键点进行旋转对齐、外扩裁剪并导出图片。
- [`train_pro.py`](train_pro.py)：定义 EfficientNet-B4 年龄/性别双分支模型，读取对齐数据，执行数据增强和 35 轮训练。

[文件校验记录](docs/original_scripts.json)保存原文件的大小与 SHA256。脚本保留原来的 Linux 路径和 CUDA 设置；运行前需要在工作副本中调整。原始训练脚本未划分独立验证集，与 C3 实验及 `age_gender/` 新代码分别保存，不能混用其结果来源。详见[原始工作流程](docs/original-workflow.md)。

## 研究内容

- **模型结构**：ImageNet 预训练 EfficientNet-B4，1792 维共享特征，分别连接年龄回归头和二分类头。
- **优化目标**：年龄 L1 损失 + 6 倍性别交叉熵，分类标签平滑系数 0.1。
- **训练策略**：AdamW、OneCycleLR、混合精度、梯度裁剪与复合数据增强。
- **对比实验**：EfficientNet-B0、EfficientNet-B4、ResNet-50、VGG-16。
- **消融实验**：去除数据增强、任务损失权重改为 1、去除人脸对齐。
- **实验追溯**：保留历史 CSV；新实验保存数据清单、文件哈希和实际训练配置。

## 现有实验记录

| 实验 | 轮数 | 最后一轮验证 MAE | 最后一轮验证分类准确率 |
| --- | ---: | ---: | ---: |
| EfficientNet-B4 主实验 | 35 | 3.4074 岁 | 98.2064% |
| EfficientNet-B4 对比基线 | 15 | 3.5445 岁 | 98.1699% |

数值来自历史训练日志，**尚未重新完成独立测试评估**。历史划分没有保存逐样本清单，原始两份同名权重也存在内容差异。本仓库采用 C3 目录中的模型与日志，并分别记录来源；不能仅凭同名文件或相同随机种子认定两次实验相同。

## 实验展示顺序

1. **模型选择（15 轮）**：EfficientNet-B4、ResNet-50、VGG-16。
2. **模型规模（15 轮）**：EfficientNet-B4 与 B0。
3. **训练策略与调参（各 15 轮）**：数据增强、损失权重、人脸对齐分别展示。
4. **最终训练（35 轮）**：单独展示选定 B4 配置的损失与验证指标。

[查看分组实验图表与结果表](docs/experiments.md)

![最终模型完整训练曲线](experiments/figures/final_training_35.png)

## 快速运行

在仓库根目录使用 Python 3.10 或更高版本，先根据 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 选择 CPU/CUDA 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m age_gender.predict --image path/to/face_crop.jpg
```

输入为已经裁剪的人脸图像；Linux/macOS 将解释器路径替换为 `.venv/bin/python`。权重文件为 `weights/efficientnet_b4_c3_epoch35.pth`，不需要额外下载 ImageNet 初始化权重。

## 训练与评估

准备有权使用的数据，按 `data/aligned/UTKFace`、`AFAD`、`imdb` 存放，图片名称以 `年龄_性别_` 开头，标签约定 0=male、1=female。

```powershell
python -m age_gender.split --data-root data/aligned --output data/manifest.csv
python -m age_gender.train --config configs/b4.json --data-root data/aligned --manifest data/manifest.csv --output runs/b4 --verify-hashes
python -m age_gender.evaluate --checkpoint runs/b4/best.pth --data-root data/aligned --manifest runs/b4/manifest.csv --split test --output runs/b4/test.json
```

新流程采用 80%/10%/10% 的训练、验证、测试划分，并将相同文件哈希的图片分入同一集合；它与历史 80%/20% 实验不同。新清单不能还原旧权重的训练成员，也不能保证同一人物或近重复图片不跨集合。正式泛化实验仍需身份信息与去重审计。

其他入口：`age_gender.preprocess`（可选人脸对齐）、`age_gender.benchmark`（模型延迟）、`age_gender.plot_results`（从 CSV 重新生成实验图）。详情见 [数据准备](docs/data.md) 和 [复现说明](docs/reproducibility.md)。

## 文件组织

`age_gender/` 是统一研究代码，`configs/` 保存新实验配置，`experiments/legacy/` 保存历史脚本，`experiments/logs/` 保存八组日志，`weights/` 保存选定模型与校验信息，`docs/` 说明实验来源、局限与检查结果，`tests/` 检查数据隔离和指标计算。

当前未对项目选择开源许可证。第三方依赖与数据遵循各自条款，详见 [来源说明](THIRD_PARTY_NOTICES.md)。仓库不包含原始数据集、私人图像或账号数据库。预测任务对应数据集中的二元标签，不代表对个人性别身份的判断。
