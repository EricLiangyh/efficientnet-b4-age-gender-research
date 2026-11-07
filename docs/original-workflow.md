# 原始数据预处理与训练流程 / Original workflow

仓库根目录的两个文件是用户提供的原始完整版本，保留原名、注释、参数与换行格式。文件大小与 SHA256 记录在 [original_scripts.json](original_scripts.json)。为了保证代码原样保存，安装与操作说明单独放在本文档。

## 1. 数据预处理：preprocess_pro.py

脚本处理三类输入：UTKFace 图片目录、AFAD 年龄/性别目录，以及本地 IMDB 清洗子集的 CSV 与图片。路径配置为：

```text
/root/my_age_project/data/
├── UTKFace/
├── AFAD-Full/
└── imdb-clean/
    ├── imdb_train_new_1024.csv
    └── imdb-clean-1024/imdb-clean-1024/
```

IMDB CSV 使用 `filename`、`age`、`gender` 字段。运行前需要核对实际数据布局和标签编码；仓库不包含这些数据集。

预处理步骤包括 InsightFace 检测与 106 点关键点、眼部关键点旋转校正、35% 外扩裁剪和 380×380 缩放。检测失败时原代码采用中心裁剪；异常处理和跳过逻辑保持原样。输出位于 `/root/my_age_project/data_aligned/` 下的 `UTKFace`、`AFAD`、`imdb` 三个子目录。

## 2. 模型训练：train_pro.py

训练脚本读取上述对齐目录，使用 EfficientNet-B4 的 ImageNet 初始化权重，连接年龄回归头和性别二分类头。原配置为 35 轮、batch size 64、最高学习率 4e-4、性别损失权重 6，以及 AdamW、OneCycleLR、混合精度和梯度裁剪。每一轮保存一个 state dictionary。

该脚本将全部可用子数据集合并训练，**没有训练/验证/测试划分，也没有验证循环**。它不是 `experiments/legacy/b4_35.py` 中的 C3 划分实验，不能将 C3 验证日志直接当作本脚本的评估成绩。

## 3. 运行前准备

原脚本面向原始 Linux/CUDA 环境，不能默认认为可以在任意机器直接运行。请先在工作副本中确认 `SOURCE_DIR`、`TARGET_DIR`、`BASE_DIR`、GPU/ONNX Runtime provider、数据目录和数据加载 worker 数量。

训练需要 PyTorch、Torchvision、Pillow 和 tqdm。预处理另需 OpenCV、NumPy、Pandas、InsightFace 以及与实际硬件匹配的 ONNX Runtime。InsightFace 预训练资产和 ImageNet 初始化资产需遵循上游条款并提前准备。可选预处理依赖见 `pyproject.toml`；原脚本指定 CUDA provider 时还需匹配的 GPU runtime。

在匹配原始配置的环境中，依次运行：

```bash
python preprocess_pro.py
python train_pro.py
```

## 三类代码的关系

| 位置 | 用途 |
| --- | --- |
| 根目录 `preprocess_pro.py`、`train_pro.py` | 本次指定的原始完整代码，逐字节保留 |
| `experiments/legacy/` | 历史实验脚本归档，包括 C3 和各对比/消融版本 |
| `age_gender/` | 后续整理的统一研究入口，与原脚本的行为差异见来源说明 |

English: The root scripts are verbatim originals. They retain their original Linux paths, CUDA settings and preprocessing behavior. Prepare the original data layout and dependencies before execution. The original training script trains on all merged samples and has no held-out evaluation loop; C3 validation results belong to a separate archived run. The unified package is a revised implementation, not a replacement for this source record.
