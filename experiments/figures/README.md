# figures：按研究问题查看图表

[完整实验页](../../docs/experiments.md) · [生成代码](../../age_gender/plot_results.py) · [日志来源](../logs/README.md)

| 阅读顺序 | 图表 | 展示内容 |
| --- | --- | --- |
| 1 | [backbone_selection_15.png](backbone_selection_15.png) | B4、ResNet-50、VGG-16，15 轮 |
| 2 | [efficientnet_scaling_15.png](efficientnet_scaling_15.png) | B4 与 B0，15 轮 |
| 3a | [augmentation_ablation_15.png](augmentation_ablation_15.png) | 使用/关闭增强，15 轮 |
| 3b | [loss_weight_ablation_15.png](loss_weight_ablation_15.png) | 性别损失权重 6 与 1，15 轮 |
| 3c | [alignment_ablation_15.png](alignment_ablation_15.png) | 对齐/未对齐，15 轮 |
| 4 | [final_training_35.png](final_training_35.png) | 最终 B4 完整 35 轮的联合损失、验证 MAE、验证准确率 |

前五张图左侧为年龄 MAE，右侧为分类准确率。最后一张单独展示完整训练过程，不将不同训练预算混在一张图上比较。

图表由历史 CSV 自动生成，未平滑、未修改测量数值。重新生成：`python -m age_gender.plot_results`。历史数据划分的限制仍然适用，详见 [来源说明](../../docs/provenance.md)。
