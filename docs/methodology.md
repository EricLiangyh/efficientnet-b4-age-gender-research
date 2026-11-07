# 数据处理、训练参数与评估说明

[返回首页](../README.md) · [实验索引](../experiments/README.md)

## 数据预处理：preprocess_pro.py

完整的目录配置、标签转换、处理细节和运行步骤见 [预处理说明](../preprocessing/README.md)。

[原始代码](../preprocessing/preprocess_pro.py)读取 UTKFace、AFAD，以及本地 IMDB 清洗子集。原始布局为：

```text
/root/my_age_project/data/
├── UTKFace/
├── AFAD-Full/
└── imdb-clean/
    ├── imdb_train_new_1024.csv
    └── imdb-clean-1024/imdb-clean-1024/
```

IMDB CSV 使用 `filename`、`age`、`gender`。预处理包含 InsightFace 检测、106 点关键点、眼部旋转校正、35% 外扩裁剪和 380×380 缩放；未检出人脸时采用中心裁剪。输出为 `data_aligned/UTKFace`、`AFAD`、`imdb`。文件名以 `年龄_性别_` 开头。

原代码旋转图片后使用原检测框裁剪，这一实现细节应在后续研究中审查。标签映射、IMDB 子集清洗和跨数据集重复也需要核对。

## 运行原始流程

脚本保留原 Linux/CUDA 路径和配置。先在工作副本中调整数据位置、输出目录、CUDA/ONNX provider 和 worker 数量，再运行：

```bash
python preprocessing/preprocess_pro.py
python train_pro.py
```

根目录 `requirements.txt` 只包含单张推理所需依赖。预处理还需要 OpenCV、NumPy、Pandas、InsightFace 和匹配硬件的 ONNX Runtime，训练还需要 tqdm。历史对比脚本可能另需 Pandas/Matplotlib。InsightFace 资产和 ImageNet 初始化权重按上游条款准备，不随仓库分发。

## 参数在哪里修改

原始训练参数在 [train_pro.py](../train_pro.py) 内，历史实验各自参数在 [对应脚本](../experiments/README.md) 内。本仓库没有第二套 JSON 配置驱动的训练框架。

| 设置 | 原始值 | 含义 |
| --- | --- | --- |
| `BATCH_SIZE` | 64 | 每批训练图片数量 |
| `EPOCHS` | 35 | 训练轮数 |
| `LR_MAX` | 4e-4 | OneCycleLR 最高学习率 |
| `GEN_LOSS_WEIGHT` | 6.0 | 性别分类损失权重 |
| AdamW `weight_decay` | 0.05 | 权重衰减 |
| `label_smoothing` | 0.1 | 分类标签平滑 |
| Dropout | 0.3 | B4 共享特征端随机失活 |
| 梯度范数上限 | 1.0 | 梯度裁剪 |

训练增强在 `get_train_transform()` 中，包括翻转、旋转、透视、颜色变化、灰度、模糊与随机擦除。模型为 ImageNet 初始化的 B4，1792 维共享特征连接年龄回归头与两个分类 logits。

## 调参与对比怎样做

先按统一预算比较 B4/ResNet-50/VGG-16，再比较 B4/B0；对增强、任务权重、对齐分别做消融。历史记录见 [分组实验结果](experiments.md)。新研究应保存明确的数据成员划分、实际参数和种子，每次改动使用独立输出目录，并同时观察年龄和分类指标。原始脚本没有验证循环，不要把训练误差当作泛化成绩。

## 如何理解评估

单张 `predict.py` 不需要真实标签，也不衡量准确率。批量效果评估需要真实年龄、真实类别，以及能够证明未用于训练的数据。

| 指标 | 含义 |
| --- | --- |
| MAE | 平均绝对年龄误差，单位岁 |
| RMSE | 均方根年龄误差，对大错误更敏感，单位岁 |
| Accuracy | 分类正确比例 |
| Precision / Recall / F1 | 分别描述各类别的分类表现，结合样本量解读 |

本仓库保留的是历史验证日志，没有提供一份能够还原原始成员划分的独立测试入口。C3 使用过 80/20 训练与验证划分；由于原始样本清单未保存，不能仅用相同随机种子重建其成员。确定参数后应在未参与选择的独立数据上最终评估。来源与限制见 [provenance.md](provenance.md)。
