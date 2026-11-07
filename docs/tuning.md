# 如何调参：从配置到实验结论

[配置入口](../configs/README.md) · [训练代码](../age_gender/train.py) · [评估说明](evaluation.md) · [历史实验图表](experiments.md)

## 先分清你在改哪套代码

| 代码 | 参数在哪里 | 怎样使用 |
| --- | --- | --- |
| 原始 `train_pro.py` | 脚本顶部常量及增强/模型函数 | 在工作副本中调整；原脚本不包含验证划分 |
| 历史 `experiments/legacy/` | 每个历史脚本自己的设置 | 阅读当时实际实验设置，与日志对应 |
| 新 `age_gender/train.py` | `configs/*.json` 与命令参数 | 推荐用于后续记录清晰的新实验 |

修改 `configs/b4.json` **不会改变根目录 `train_pro.py` 的行为**。`evaluate.py` 负责测量结果，不负责更新模型参数。

## 按研究问题做实验

| 问题 | 比较设置 | 需要保持一致的条件 |
| --- | --- | --- |
| 选择哪个骨干 | B4 / ResNet-50 / VGG-16，各 15 轮 | 数据划分、训练预算、输入大小、优化设置 |
| B4 规模是否值得 | B4 / B0，各 15 轮 | 同一份清单和评价指标；另测参数量/耗时 |
| 增强有没有帮助 | 完整增强 / 关闭增强 | 同一 B4 配置和清单 |
| 两任务怎样平衡 | 性别损失权重 6 / 1 | 其他训练设置相同，同时观察年龄与分类指标 |
| 人脸对齐有没有帮助 | 对齐 / 未对齐 | 对应同一组原始样本与相同成员划分 |
| 最终训练是否收敛 | 选定配置训练 35 轮 | 用验证集观察收敛，测试集留到最后 |

以上是后续实验设计方法；历史结果是单次运行记录，尚不能自动证明所有条件完全受控。

## 实际操作示例：比较任务权重

准备好同一份 `data/manifest.csv` 后，分别运行：

```powershell
python -m age_gender.train --config configs/b4_15.json --data-root data/aligned --manifest data/manifest.csv --output runs/b4_weight6_15 --verify-hashes
python -m age_gender.train --config configs/b4_gender_weight_1_15.json --data-root data/aligned --manifest data/manifest.csv --output runs/b4_weight1_15 --verify-hashes
```

查看两个输出目录的 `training.csv`，比较 `val_mae` 和 `val_accuracy`，不要只看训练误差。新日志的 accuracy 为 0～1；历史日志的 `val_acc` 为百分比。

要尝试新的权重值，复制配置为新文件，只修改 `gender_loss_weight`，并给运行使用新的输出目录。例如设为 3 是一个**待运行的新试验**，不是本仓库已经测得的有效设置。不要覆盖原始配置和历史日志。

## 常见参数改动的含义

- `lr_max`：改变最高学习率。需要实际观察验证曲线，不能只根据训练下降速度判断。
- `batch_size`：影响内存和优化行为，调整后应作为不同实验记录；显存不足可以先降低它。
- `epochs`：改变训练预算，比较模型时先保持一致。命令行 `--epochs` 会覆盖 JSON 值。
- `gender_loss_weight`：调整两任务相对权重，同时记录两个任务的结果。
- `augmentation`：只控制新代码中的训练随机增强；验证/测试使用确定性的变换。
- `image_size`：改变输入分辨率，也会改变计算代价；不能将不同分辨率的差异只归因于骨干。
- `dropout`：当前新实现作用于 EfficientNet/ResNet 的特征端；VGG 保留原分类器 dropout，调此字段不会改变它。
- `seed`：用于新训练随机数。若需要更改数据划分，须单独运行 `split.py --seed` 并保存新的清单。

每组多种子比较最好固定同一数据清单，仅改变训练种子；如果同时换划分和训练种子，应分别记录并解释。

## 怎样判断和保存结果

每次新训练输出 `run.json`、`manifest.csv`、`training.csv`、`best.pth`、`last.pth`。`best.pth` 按验证年龄 MAE 选择；`last.pth` 是最后一轮，二者可能不同。当前权重不保存优化器状态，未实现中断恢复。

调参时比较验证结果，最终方案确定后，再按 [评估指南](evaluation.md) 评估测试集。报告实际配置、训练轮数、数据清单哈希、随机种子、两个任务指标和运行环境。多次使用测试集挑参数，会破坏它作为最终独立评估的作用。
