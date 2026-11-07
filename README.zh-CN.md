# 基于 EfficientNet-B4 的人脸年龄与性别预测研究

**围绕人脸年龄回归与性别标签分类的个人模型研究项目。** 研究共享 EfficientNet-B4 特征下，骨干选择、人脸对齐、数据增强和任务损失加权对两个任务的影响。

[English](README.md) · [实验结果](docs/experiments.md) · [模型卡](docs/model-card.md)

## 为什么开展这项研究

面向图像客群属性分析，年龄估计与属性分类往往需要同时完成，而姿态、图片质量和数据来源差异会影响预测。项目尝试用共享视觉特征联合处理两个任务，通过对比与消融实验分析具体处理策略的作用，重点是已有架构的任务适配与实验研究。

## 我的研究工作

- 整理 UTKFace、AFAD 和本地 IMDB 子集，构建标签转换、关键点旋转校正、外扩裁剪的数据预处理流程。
- 将 ImageNet 预训练 EfficientNet-B4 适配为共享骨干、年龄回归头与性别分类头的多任务模型。
- 开展骨干比较和增强、损失权重、对齐消融，保留八组历史训练日志与对应实现。
- 分析收敛过程及两任务表现，整理选定权重和最小单张人脸预测入口。

研究价值体现在数据处理、任务建模、训练策略和实验分析。EfficientNet 及预训练组件归属原作者，不将其表述为自主提出的新骨干或领域领先成果。

## 先看这三个入口

| 顺序 | 入口 | 重点内容 |
| --- | --- | --- |
| **1. 数据预处理** | [preprocessing/](preprocessing/README.md) | 完整原始脚本、目录配置、标签转换、对齐裁剪 |
| **2. 模型训练** | [training/](training/README.md) | 完整原始实现、模型结构、增强、损失和参数 |
| **3. 训练好的模型** | [weights/](weights/README.md) | C3 权重、预测命令和输入要求 |

### 训练代码与现有模型怎样对应

| 材料 | 训练方式 | 与现有权重的关系 |
| --- | --- | --- |
| [原始 train_pro.py](training/train_pro.py) | 合并全部数据训练，没有独立验证循环 | 原始实现完整保留；属于另一历史训练版本 |
| [C3 b4_35.py](experiments/legacy/b4_35.py) | 按图片进行 80/20 训练/验证划分，35 轮 | 按来源目录关联 [当前权重](weights/efficientnet_b4_c3_epoch35.pth) 与 [C3 日志](experiments/logs/b4_35.csv) |

原始文件未经修改，文件校验信息已保存。但历史逐样本划分清单未保留，目录关联和结构兼容不能独立证明完整训练来源。详见 [来源记录](docs/provenance.md)。

## 模型方法

```text
对齐 RGB 人脸，380×380 → ImageNet 归一化 → EfficientNet-B4
                                           ↓ 共享 1792 维特征
                                           ├── 年龄回归：1 个连续值
                                           └── 性别标签分类：2 个 logits
```

联合目标为 `年龄 L1 损失 + 6 × 分类交叉熵`，标签平滑 0.1；使用 AdamW、OneCycleLR、混合精度与复合增强。实现涉及 PyTorch/Torchvision、InsightFace、OpenCV 和 Pillow，具体设置见 [方法说明](docs/methodology.md)。

## 实验发现

下表来自**各组 15 轮实验的最后一轮验证日志**，MAE 越低越好：

| 研究问题 | 历史记录中的观察 |
| --- | --- |
| 骨干选择 | B4 的 MAE 为 **3.5445 岁**；ResNet-50 为 3.8423，VGG-16 为 4.1115 |
| B4 与 B0 | B4 的 MAE 为 **3.5445 岁**，B0 为 3.6903 |
| 数据增强 | 不增强时 MAE 为 3.6520，完整增强时为 **3.5445** |
| 任务损失权重 | 性别损失权重从 1 改为 6，分类准确率由 97.8283% 变为 **98.1699%**；年龄 MAE 仅小幅变化，3.5490 → 3.5445 |
| 人脸对齐 | 分类准确率由 97.4825% 变为 **98.1699%**，MAE 由 3.6101 变为 3.5445 |

单独的 **C3 35 轮训练记录**最终为 **MAE 3.4074 岁、分类准确率 98.2064%**。训练预算不同，因此不与 15 轮实验混为同一组对比。

这些是单次历史验证观察，不是本次重新测得的独立测试成绩，也不足以证明统计显著性或完全受控的因果效果。

[分组实验图与 CSV 来源](docs/experiments.md) · [实验代码索引](experiments/README.md)

![最终 C3 训练](experiments/figures/final_training_35.png)

## 从下载到预测

需要 Python、Git 和兼容的 PyTorch/Torchvision 环境。仓库仍为私有，克隆时需要有访问权限的账号。

```powershell
git clone https://github.com/EricLiangyh/efficientnet-b4-age-gender-research.git
cd efficientnet-b4-age-gender-research
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe weights/predict.py --image "path/to/face_crop.jpg" --device cpu
```

Linux/macOS 将 `.\.venv\Scripts\python.exe` 换为 `.venv/bin/python`。使用 GPU 时先准备匹配的 PyTorch/CUDA 环境。预测直接加载仓库权重，不另外下载 ImageNet 初始化参数。

### 已授权的照片示例

<img src="weights/example_face.png" alt="用户授权展示的示例照片，保留未裁剪原图" width="180" />

照片由项目所有者提供，并已确认可用于本仓库展示；文件保持原样，未修改人物或图像内容。原图包含上半身和背景，推理前需裁出一张完整人脸，保留少量额头、下巴边缘，将工作副本保存为本地 `face_crop.jpg`。不必手动缩放到 380×380。

本照片未附实测预测结果或真实标签。此图用于说明输入准备，不作为准确率证据。展示授权不代表访客获得通用转载许可。

准备好裁剪图后运行：



```powershell
.\.venv\Scripts\python.exe weights/predict.py --image "path/to/face_crop.jpg" --device cpu
```

图片应以一张完整人脸为主体，保留适当额头、下巴边缘；不要直接输入整张多人合照。尺寸由程序自动调整，不含自动人脸检测。详见 [模型使用说明](weights/README.md)。

## 研究边界与下一步

仓库保留原始研究实现、历史实验依据与模型推理，不是开箱即用的完整训练评估框架。训练和预处理还需单独准备依赖、数据及原 Linux/CUDA 路径。尚未重新完成全数据评估、原始划分还原或身份隔离测试。

下一步优先考虑身份与近重复审查、固定数据清单、独立外部测试集、多随机种子和分组误差分析。二元数据集标签不等同于个人性别身份，更多使用限制见 [模型卡](docs/model-card.md)。

```text
preprocessing/   原始预处理与配置说明
training/        原始训练与参数说明
weights/         选定权重、预测代码和输入输出示例
experiments/     历史脚本、八组日志与六张图
docs/            方法、结果、模型卡与来源
```

作者：**EricLiangyh（梁逸恒）**。当前未选择项目许可证；代码、数据和预训练模型遵循各自适用条款，详见 [来源与许可说明](THIRD_PARTY_NOTICES.md)。仓库不分发训练数据集或私人照片。
