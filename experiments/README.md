# experiments：模型选择与调参的实验依据

[返回首页](../README.md) · [按研究步骤看图表](../docs/experiments.md) · [如何开展新调参实验](../docs/tuning.md)

读完原始预处理、训练代码和权重后，可以在这里深入了解“为什么选 B4、为什么使用这些训练策略”。

## 推荐阅读顺序

| 顺序 | 问题 | 历史脚本 | 历史记录 |
| --- | --- | --- | --- |
| 1 | B4、ResNet-50、VGG-16 选择哪个 | [脚本导航](legacy/README.md) 中的三组 15 轮实验 | `logs/b4_15.csv`、`resnet50_15.csv`、`vgg16_15.csv` |
| 2 | EfficientNet 选择 B4 还是 B0 | B4/B0 两组 15 轮实验 | `logs/b4_15.csv`、`b0_15.csv` |
| 3 | 增强、任务权重、对齐的影响 | 三类消融分别比较 | `logs/b4_no_augmentation_15.csv`、`b4_gender_weight_1_15.csv`、`b4_no_alignment_15.csv` |
| 4 | 最终模型训练过程怎样 | [b4_35.py](legacy/b4_35.py) | [完整 35 轮日志](logs/b4_35.csv) |

每组比较的图与末轮结果表统一在 [实验结果页](../docs/experiments.md)。这是按研究问题组织的阅读顺序，不表示已经证明原实验的实际执行时间顺序。

## 各目录和汇总文件做什么

| 位置 | 内容 | 怎样使用 |
| --- | --- | --- |
| [legacy/](legacy/README.md) | 历史训练及辅助脚本 | 核对历史方法和参数；脚本有原环境路径，不能默认直接运行 |
| [logs/](logs/README.md) | 八组逐轮 CSV 日志 | 查看训练损失、验证误差与分类准确率 |
| [figures/](figures/README.md) | 从上述 CSV 生成的六张图 | 每张图只回答一组研究问题 |
| [summary.json](summary.json) | 绘图程序生成的各实验最后一轮摘要 | 用于读取汇总；它不是额外一组测量 |
| [historical_results.json](historical_results.json) | 初次整理保留的历史末轮记录及来源 | 作为辅助来源索引，不是独立测试结果 |

重画图表：从仓库根目录运行 `python -m age_gender.plot_results`。不要手动修改图中的数值来替代真实日志。

历史日志记录的是验证过程，不应自动称为独立测试成绩。数据成员划分和版本限制详见 [来源说明](../docs/provenance.md)。
