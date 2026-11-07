# 数据集预处理 / Dataset preprocessing

[返回项目首页](../README.md) · [查看完整代码：preprocess_pro.py](preprocess_pro.py) · [下一步：模型训练](../training/README.md)

本目录只包含原始预处理脚本和这份说明。脚本逐字节保留，未修改算法、参数或注释；文件校验记录见 [original_scripts.json](../docs/original_scripts.json)。

## 这份代码做什么

将 UTKFace、AFAD 和本地 IMDB 清洗子集整理为统一大小的人脸图片，供年龄/性别模型训练读取。

```text
原始图片与标签
    ↓ 扫描三个数据源，统一年龄/性别文件名
InsightFace 人脸检测 + 106 点关键点
    ↓ 选检测置信度最高的人脸
根据眼部关键点计算角度，旋转原图
    ↓ 按检测框四周各外扩 35% 裁剪
缩放到 380 × 380
    ↓
data_aligned/UTKFace、AFAD、imdb
```

没有检测到人脸时，采用中心正方形裁剪后缩放。预处理只导出图片，不进行训练/验证/测试划分；随机数据增强在训练脚本中执行。

## 1. 配置输入和输出位置

原脚本顶部的配置：

```python
SOURCE_DIR = "/root/my_age_project/data"
TARGET_DIR = "/root/my_age_project/data_aligned"
IMG_SIZE = 380
PADDING_RATIO = 0.35
```

| 参数 | 作用 | 修改时注意 |
| --- | --- | --- |
| `SOURCE_DIR` | 三个原始数据源的共同根目录 | 使用实际数据所在位置 |
| `TARGET_DIR` | 预处理图片输出根目录 | 训练代码的 `DATA_DIR` 应指向这里 |
| `IMG_SIZE` | 输出图片宽、高，默认 380 | 当前 B4 训练使用 380×380，改变后需同步检查训练配置 |
| `PADDING_RATIO` | 检测框每一侧的扩展比例 | 0.35 表示左/右各加框宽的 35%，上/下各加框高的 35%；未遇边界时约为原框的 1.7 倍宽高 |
| `DIR_MAP` | 各数据源在输入根目录下的相对路径 | 实际解压层级可能不同，必须核对 |

脚本没有命令行路径参数；请在自己的工作副本中修改这些常量。输出目录不要与原始数据目录混用。

## 2. 按原配置准备目录

```text
/root/my_age_project/data/
├── UTKFace/
│   └── 25_0_0_example.jpg
├── AFAD-Full/
│   └── 28/
│       ├── 111/
│       └── 112/
└── imdb-clean/
    ├── imdb_train_new_1024.csv
    └── imdb-clean-1024/
        └── imdb-clean-1024/
            └── 图片路径由 CSV 的 filename 字段指定
```

这里只说明原脚本读取的布局，不随仓库分发数据集。数据使用条款见 [第三方来源](../THIRD_PARTY_NOTICES.md)。

## 3. 标签如何转换

| 来源 | 原脚本读取方式 | 输出文件名 |
| --- | --- | --- |
| UTKFace | 扫描 `UTKFace` 目录中的 `.jpg`、`.png` 文件 | 保留原文件名；训练依靠开头的年龄和性别字段 |
| AFAD | 遍历数字年龄目录下的 `111`、`112` 子目录 | `{age}_{gender}_afad_{原文件名}`；111→0，112→1 |
| IMDB 本地子集 | 读取 CSV 的 `filename`、`age`、`gender` 三列 | `{整数年龄}_{gender}_imdb_{原文件名的 basename}` |

性别标签约定为 0=male、1=female。IMDB 原代码将 `F`、`FEMALE`、`1.0`、`1` 转为 1，其他值都转为 0；因此运行前必须核对 CSV 的标签语义和缺失值，不能默认认为所有其他值都是有效 male 标签。IMDB 年龄通过 `int()` 转换。

## 4. 人脸处理的实现细节

1. 用 OpenCV 读取图片。
2. InsightFace 启用 `detection` 和 `landmark_2d_106`，检测输入大小为 640×640。
3. 多张人脸时，只取检测置信度最高的一张，并非批量导出全部人脸。
4. 原实现取 106 点关键点中的索引 39 和 89，计算两点连线的旋转角度，以其中点旋转整幅图片。
5. 使用原检测框四周外扩 `PADDING_RATIO` 后裁剪，裁剪范围限制在图片边界内。
6. 缩放到 `IMG_SIZE × IMG_SIZE`，写入输出目录。

原实现旋转图片后未同步变换检测框坐标，这是保留的算法行为，后续修改需要单独验证。裁剪为空时，代码会将原图直接缩放；未检测到人脸时则走中心裁剪。

## 5. 运行环境和命令

脚本需要 `opencv-python`、`pandas`、`numpy`、`insightface`、`tqdm`，以及匹配实际设备的 ONNX Runtime。根目录 `requirements.txt` 是单张推理的最小依赖清单，不能据此认为预处理依赖已全部安装。

原文件指定：

```python
providers=['CUDAExecutionProvider']
app.prepare(ctx_id=0, det_size=(640, 640))
```

这是原始 GPU 环境设置，需要可用的 CUDA provider 和相应模型资产。CPU 工作副本可改为 `CPUExecutionProvider`，并将 `ctx_id` 改为 `-1`，运行前仍需确认 InsightFace 模型和运行库可用。InsightFace 资产遵循其上游条款。

在仓库根目录执行：

```bash
python preprocessing/preprocess_pro.py
```

## 6. 输出与训练如何连接

```text
/root/my_age_project/data_aligned/
├── UTKFace/
├── AFAD/
└── imdb/
```

训练脚本原配置为 `BASE_DIR = "/root/my_age_project"`、`DATA_DIR = f"{BASE_DIR}/data_aligned"`。如果改了预处理输出位置，训练读取目录也要同步调整。然后按 [方法说明](../docs/methodology.md) 继续训练。

## 7. 运行结果需要检查什么

- 已存在的目标文件会直接跳过；如果修改对齐参数后想重新处理，请使用新的输出目录，避免混用旧结果。
- 不存在的数据源会被跳过；任务数量为零时先核对输入目录及 CSV。
- 读取失败的图片会跳过，部分处理异常也会静默跳过；程序完成不代表每张原图均成功导出。
- `app.get(img)` 在内部处理的 `try` 外，检测阶段异常不保证被静默跳过；输出计数也没有逐项成功报告。
- AFAD/IMDB 文件名转换存在同名覆盖的可能，尤其是 IMDB 不同子目录的同名文件；应核对原始数量、输出数量与文件名唯一性。
- 检查若干年龄段和数据源的样例，确认标签、旋转和裁剪合理，再进入训练。

English: This folder preserves the original preprocessing script unchanged. Configure source/output paths and dataset layout inside the script, prepare InsightFace and the appropriate ONNX Runtime provider, then run `python preprocessing/preprocess_pro.py` from the repository root. Output folders feed the training script's `DATA_DIR`; preprocessing does not create evaluation splits.
