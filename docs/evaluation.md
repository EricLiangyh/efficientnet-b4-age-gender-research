# evaluate.py：模型效果评估指南

[返回代码导航](../age_gender/README.md) · [查看代码](../age_gender/evaluate.py) · [调参指南](tuning.md)

## 它是什么，不需要猜文件名

`evaluate.py` 用于**评估已经训练好的模型**。它读取有真实年龄和分类标签的一批图片，将预测值与真实标签比较，汇总误差和分类指标。它不进行梯度更新，不负责训练或自动调参。

| 操作 | 使用的代码 | 是否需要真实标签 | 是否修改模型 |
| --- | --- | --- | --- |
| 训练 | `age_gender/train.py` | 需要 | 是 |
| 单张预测 | `age_gender/predict.py` | 不需要 | 否 |
| 批量效果评估 | `age_gender/evaluate.py` | 需要 | 否 |
| 推理耗时测量 | `age_gender/benchmark.py` | 不需要，使用合成张量 | 否 |

## 需要准备什么

1. 权重文件：新训练产生的 `best.pth` 或 `last.pth`，或明确标注来源的历史权重。
2. 图片根目录：与清单中相对路径组合后，能够找到实际图片。
3. 数据划分清单：CSV 含 `path,age,gender,source,sha256,split`。评估程序不临时重新随机划分数据。

程序默认核对清单 SHA256 是否与新训练权重保存的一致，并核对每张被评估图片的文件 SHA256。它检查记录一致性，但不能自动证明人物身份不重叠或数据标注正确。

## 推荐用法：评估新实验

先按 [代码导航](../age_gender/README.md) 完成新训练，然后在仓库根目录运行：

```powershell
python -m age_gender.evaluate --checkpoint runs/b4/best.pth --data-root data/aligned --manifest runs/b4/manifest.csv --split test --device cpu --batch-size 32 --output runs/b4/test.json
```

| 参数 | 用途 |
| --- | --- |
| `--checkpoint` | 要评估的权重，必填 |
| `--data-root` | 图片根目录，必填 |
| `--manifest` | 已保存的数据清单，必填 |
| `--split` | 评估 `val` 或 `test`，默认 `test` |
| `--device` | 默认 `cpu`；有匹配环境时可指定 `cuda` |
| `--batch-size` | 每批评估图片数，默认 32；它不是重新训练参数 |
| `--output` | 指标 JSON 路径，默认 `runs/evaluation.json` |
| `--allow-unverified-manifest` | 明确允许缺少匹配训练清单的历史权重进行探索性评估；不会证明测试独立性 |

## 输出怎样阅读

指定 `--output runs/b4/test.json` 时，会同时得到：

- `runs/b4/test.json`：总体指标、各类指标、权重/清单哈希及来源提示。
- `runs/b4/test.predictions.csv`：逐张图片的路径、真实年龄、预测年龄、真实标签和预测标签，便于分析失败案例。

| 字段 | 解释 |
| --- | --- |
| `age_mae_years` | 平均绝对误差，单位为岁，越低越好 |
| `age_rmse_years` | 均方根误差，单位为岁，对较大错误更敏感 |
| `gender_accuracy` | 分类准确率，保存为 0～1，例如 0.98 表示 98% |
| `confusion_matrix` | 行是真实类别、列是预测类别，顺序为 0、1 |
| `classes` | 各类别的 precision、recall、F1 与实际样本量 |
| `manifest_matches_checkpoint` | 当前清单哈希是否匹配权重记录；并非“已证明没有任何泄漏” |
| `independence_note` | 对数据成员来源和独立性限制的解释 |

历史 CSV 的 `val_acc` 使用 0～100 百分比，新评估 JSON 的 `gender_accuracy` 使用 0～1 比例；比较时先统一单位。没有真实样本的类别不应仅凭零值指标得出结论。

## 为什么历史 C3 权重会被默认拒绝

历史 C3 权重没有保存训练样本清单。因此直接用新清单评估，会出现清单来源未核实的提示。若明确只做探索性测量，可以执行：

```powershell
python -m age_gender.evaluate --checkpoint weights/efficientnet_b4_c3_epoch35.pth --data-root data/aligned --manifest data/manifest.csv --split test --allow-unverified-manifest --output runs/c3_exploratory.json
```

这不会还原历史划分，也不能把这些图片自动认定为模型从未见过的测试集。结果会保留来源未核实的标记。要报告新的独立测试成绩，应使用能够证明未参与训练的外部数据，或完整记录划分后重新训练。

## 评估与调参怎样连接

先用训练集拟合，在验证集上比较参数；确定方案后再对预留测试集评估。新训练按**验证年龄 MAE 最小**保存 `best.pth`，并非按两个任务的综合最优选择。若研究目标改为更侧重分类，需要显式设计新的选择标准并记录，不能事后从测试成绩挑最优模型。
