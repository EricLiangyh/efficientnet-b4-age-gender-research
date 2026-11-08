# 已训练模型 / Trained model

[返回首页](../README.md) · [模型卡](../docs/model-card.md) · [来源说明](../docs/provenance.md)

| 文件 | 用途 |
| --- | --- |
| [efficientnet_b4_c3_epoch35.pth](efficientnet_b4_c3_epoch35.pth) | 选定的 C3 第 35 轮 B4 参数，约 71 MB |
| [predict.py](predict.py) | 加载同目录权重，预测一张已裁剪人脸图片 |
| [manifest.json](manifest.json) | 文件大小、SHA256，以及另一历史权重的区别 |

权重按目录来源关联 [C3 训练脚本](../experiments/legacy/b4_35.py) 与 [日志](../experiments/logs/b4_35.csv)。原始全数据 `training/train_pro.py` 是另一版本，不能直接认定为本权重的生成代码。

## 如何使用

准备兼容的 PyTorch/Torchvision 环境后，在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python weights/predict.py --image "path/to/face_crop.jpg" --device cpu
```

试用真实图片时，将 `path/to/face_crop.jpg` 替换为自己有权使用的人脸裁剪图片路径；路径包含空格时加双引号。默认使用 CPU，在配置好 CUDA 的环境中可指定 `--device cuda`。默认权重按脚本所在目录查找；使用其他同结构权重时，可通过 `--checkpoint` 指定文件。

## “已裁剪的人脸图片”是什么意思

输入图片应以**一张人脸为主体**，保留完整脸部和适当的额头、下巴边缘，尽量扶正，避免大量背景。不是直接输入整张多人合照或监控画面；这份预测代码不负责定位人脸。

可以使用已有的人脸裁剪图、手动裁剪的单人脸图，或 [预处理脚本](../preprocessing/README.md) 导出的图片。不必提前将尺寸改成 380×380，脚本会自动缩放并归一化；手动裁剪不保证与训练时的对齐方式完全一致，可能影响效果。

输出为年龄估计、`gender_label`（0=male，1=female）和按 `[male, female]` 排列的 `label_probabilities`。概率未经校准，单张预测不衡量准确率，也不判断个人性别身份。

`predict.py` 仅定义加载这份权重所需的 B4 结构，以 `weights_only=True` 严格加载参数，不额外下载 ImageNet 权重。它不是另一套训练实现。

No project-wide license is selected. Dataset and pretraining terms apply separately. Successful loading and finite outputs do not establish accuracy on unseen data.
