"""
年龄和性别预测模型训练脚本 (C4实验 - EfficientNet-B0对照)
==========================================================
实验目的: 验证EfficientNet-B4相比B0的性能提升

模型架构: EfficientNet-B0 (双分支: 年龄回归 + 性别分类)
与C3(EfficientNet-B4)的区别:
  - 使用B0而非B4(参数量更少,速度更快,但精度可能更低)
  - B0特征维度: 1280维 vs B4: 1792维
  
训练策略: 
  - ImageNet预训练权重迁移学习
  - 混合精度训练(AMP)加速
  - OneCycleLR学习率调度
  - 数据增强提升泛化能力
  - 多数据集融合(UTKFace + AFAD + IMDB)
  
输出:
  - 每轮保存模型权重
  - 自动生成训练曲线图
  - 保存CSV训练日志
"""

# ==========================================
# 导入依赖库
# ==========================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import pandas as pd          # 用于保存CSV训练日志
import matplotlib.pyplot as plt  # 用于绘制并保存训练曲线图

# ==========================================
# 1. 核心配置参数 (C4实验 - EfficientNet-B0)
# ==========================================
# 项目根目录(所有路径基于此)
BASE_DIR = "/root/my_age_project"
# 对齐后的人脸数据目录(由preprocess_pro.py生成)
DATA_DIR = f"{BASE_DIR}/data_aligned" 
# 模型权重和训练日志保存目录(B0实验独立保存)
SAVE_DIR = f"{BASE_DIR}/checkpoints_pro_v1_b0"

# 训练超参数
BATCH_SIZE = 64           # 批次大小: 每次训练64张图片
EPOCHS = 15               # 训练轮数: 15轮(对照实验)
LR_MAX = 4e-4             # 最大学习率: 0.0004
GEN_LOSS_WEIGHT = 6.0     # 性别损失权重: 性别损失×6(与C3保持一致)

# 设备选择: 优先使用GPU,否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存目录(如果不存在)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/plots", exist_ok=True)  # 专门存放训练曲线图的子目录

# ==========================================
# 2. 模型定义: EfficientNet-B0双分支架构 (C4实验)
# ==========================================
class AgeGenderModel(nn.Module):
    """
    年龄和性别预测模型 - C4实验使用EfficientNet-B0
    
    与C3(EfficientNet-B4)的对比:
    ┌─────────────────────────────────────────────────────┐
    │ 特性          | EfficientNet-B0 | EfficientNet-B4   │
    │ ─────────────────────────────────────────────────── │
    │ 参数量        | ~5.3M          | ~19M              │
    │ 特征维度      | 1280维         | 1792维            │
    │ 计算量(FLOPs) | ~0.39B         | ~4.2B             │
    │ 速度          | 快             | 慢                │
    │ 精度          | 较低           | 较高              │
    └─────────────────────────────────────────────────────┘
    
    实验目的: 验证模型复杂度对性能的影响
    """
    def __init__(self):
        super().__init__()
        # 加载EfficientNet-B0预训练模型(ImageNet权重)
        # B0是最小的EfficientNet变体,参数量少,训练快
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 获取原始分类层的输入特征维度(EfficientNet-B0为1280)
        in_feats = self.backbone.classifier[1].in_features
        
        # 替换原始分类层: 移除ImageNet的1000类分类头
        # 保留Dropout层(防止过拟合),直接输出特征向量
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),  # 30%的神经元随机失活
            nn.Identity()                      # 直接输出1280维特征
        )
        
        # 年龄预测头: 回归任务(输出连续值)
        # 输入: 1280维特征 → 输出: 1个年龄值
        self.age_head = nn.Linear(in_feats, 1)    
        
        # 性别预测头: 分类任务(输出二分类概率)
        # 输入: 1280维特征 → 输出: 2个类别(男/女)的logits
        self.gender_head = nn.Linear(in_feats, 2) 

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像张量 [batch_size, 3, 380, 380]
        
        返回:
            (age_pred, gender_pred): 
                - age_pred: 年龄预测 [batch_size, 1]
                - gender_pred: 性别预测 [batch_size, 2]
        """
        # 通过主干网络提取特征
        feat = self.backbone(x)  # feat形状: [batch_size, 1280]
        # 双分支同时预测
        return self.age_head(feat), self.gender_head(feat)

# ==========================================
# 3. 数据集解析类: 读取对齐后的人脸图像
# ==========================================
class AlignedDataset(Dataset):
    """
    自定义数据集类,用于加载预处理后的人脸图像
    
    文件名格式: {年龄}_{性别}_{原始文件名}.jpg
    例如: 25_0_face123.jpg (25岁,男性)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(root_dir):
            self.files = []
        else:
            self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        parts = file_name.split('_')
        try:
            age = float(parts[0])
            gender = int(parts[1])
            img_path = os.path.join(self.root_dir, file_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, torch.tensor(age), torch.tensor(gender)
        except Exception:
            return self.__getitem__(torch.randint(0, len(self.files), (1,)).item())

# ==========================================
# 4. 数据增强策略
# ==========================================
def get_train_transform():
    """
    训练集数据增强流水线
    
    增强策略(按执行顺序):
    1. Resize: 统一缩放到380×380
    2. RandomHorizontalFlip: 50%概率水平翻转
    3. RandomRotation: ±15度随机旋转
    4. RandomPerspective: 50%概率透视变换
    5. ColorJitter: 随机调整亮度/对比度/饱和度/色调
    6. RandomGrayscale: 10%概率转灰度
    7. GaussianBlur: 20%概率高斯模糊
    8. ToTensor: 转换为PyTorch张量
    9. Normalize: ImageNet标准化
    10. RandomErasing: 30%概率随机遮挡
    """
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    ])

def get_val_transform():
    """验证集数据变换(无随机增强,保证评估准确)"""
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ==========================================
# 5. 训练主循环: 多任务联合训练流程
# ==========================================
class TransformSubset(Dataset):
    """
    包装random_split返回的Subset,动态应用不同的transform
    
    为什么需要这个类?
    - random_split返回的Subset没有transform属性
    - 训练集和验证集需要不同的transform(增强 vs 无增强)
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img, age, gender = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, age, gender

def train():
    """
    主训练函数 - C4实验(EfficientNet-B0)
    
    训练流程:
    1. 加载三个数据集(UTKFace, AFAD, IMDB)
    2. 划分80%训练集和20%验证集
    3. 初始化模型、优化器、学习率调度器
    4. 循环训练EPOCHS轮
    5. 每轮保存模型权重和训练指标
    6. 训练结束后生成可视化图表
    """
    # ==========================================
    # 步骤1: 数据准备 (加载、合并、划分数据集)
    # ==========================================
    # 指定要加载的三个数据集子集名称(UTKFace、AFAD、IMDB)
    subsets = ["UTKFace", "AFAD", "imdb"] 
    
    # 初始化空列表,用于存储加载的数据集对象
    dataset_list = []
    
    # 遍历每个数据集目录,逐个加载
    for s in subsets:
        # 拼接数据集完整路径,例如: /root/my_age_project/data_aligned/UTKFace
        path = os.path.join(DATA_DIR, s)
        
        # 检查目录是否存在(防止路径错误导致崩溃)
        if os.path.exists(path):
            # 加载数据集(不应用transform,后续通过TransformSubset动态应用)
            # transform=None表示此时只解析文件,不做图像变换
            dataset_list.append(AlignedDataset(path, transform=None))
            
    # 检查是否成功找到数据(防止空数据集启动训练)
    if not dataset_list:
        # 打印错误提示,显示搜索路径
        print(f"❌ 错误：在 {DATA_DIR} 未找到任何对齐后的数据！")
        # 终止训练函数,返回None
        return

    # 合并多个数据集为一个大集合(ConcatDataset将多个Dataset串联)
    # 例如: UTKFace(20000张) + AFAD(30000张) + IMDB(50000张) = 100000张
    full_dataset = ConcatDataset(dataset_list)
    
    # 计算数据集总样本数
    total_size = len(full_dataset)
    
    # 计算训练集样本数(80%)
    train_size = int(0.8 * total_size)
    
    # 计算验证集样本数(剩余20%)
    val_size = total_size - train_size
    
    # 随机划分训练集和验证集(80/20比例)
    # generator=torch.Generator().manual_seed(42)设置固定随机种子,保证每次划分结果相同(可重复性)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # 使用TransformSubset包装训练集子集,应用数据增强transform
    # get_train_transform()返回包含翻转、旋转、颜色抖动等增强的变换流水线
    train_dataset = TransformSubset(train_subset, get_train_transform())  # 训练集: 有数据增强
    
    # 使用TransformSubset包装验证集子集,应用基础transform(无增强)
    # get_val_transform()返回仅包含Resize和Normalize的基础变换
    val_dataset = TransformSubset(val_subset, get_val_transform())        # 验证集: 无数据增强

    # 创建训练集DataLoader(用于批量加载数据)
    # batch_size=BATCH_SIZE: 每次加载64张图片
    # shuffle=True: 每个epoch随机打乱训练集顺序(防止模型记忆顺序)
    # num_workers=8: 使用8个子进程并行加载数据(加速数据读取)
    # pin_memory=True: 将数据预分配到GPU专用内存(加速CPU到GPU的数据传输)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    
    # 创建验证集DataLoader
    # shuffle=False: 验证集不需要打乱(保持固定顺序便于对比)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # ==========================================
    # 步骤2: 模型与优化器初始化 (构建训练环境)
    # ==========================================
    # 创建AgeGenderModel模型实例并移动到GPU设备
    # .to(DEVICE)将模型所有参数从CPU内存复制到GPU显存
    model = AgeGenderModel().to(DEVICE)
    
    # 创建AdamW优化器(带权重衰减的正则化优化器)
    # model.parameters(): 传入模型所有可训练参数
    # lr=LR_MAX: 设置初始学习率为0.0004
    # weight_decay=0.05: 权重衰减系数,惩罚过大的权重值,防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.05) 
    
    # 创建OneCycleLR学习率调度器(动态调整学习率)
    # 学习率变化策略: 线性增长到LR_MAX(前20%步数) → 余弦衰减到接近0(后80%步数)
    # max_lr=LR_MAX: 最大学习率为0.0004
    # steps_per_epoch=len(train_loader): 每个epoch的步数(=训练集样本数/batch_size)
    # epochs=EPOCHS: 总共15个epoch
    # pct_start=0.2: 前20%的步数用于warmup(学习率线性增长阶段)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, 
                                             steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2) 
    
    # 定义年龄损失函数: L1Loss(MAE,平均绝对误差)
    # MAE = |预测年龄 - 真实年龄|,对异常值不敏感
    criterion_age = nn.L1Loss()
    
    # 定义性别损失函数: CrossEntropyLoss(交叉熵,带标签平滑)
    # label_smoothing=0.1: 将硬标签[1,0]软化为[0.9,0.1],防止模型过度自信,提升泛化能力
    criterion_gen = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 创建混合精度训练GradScaler(使用FP16半精度加速训练)
    # GradScaler动态缩放损失值,防止FP16精度丢失小梯度
    scaler = GradScaler('cuda')

    # 初始化空列表,用于记录每个epoch的训练和验证指标
    history = []

    # 打印训练启动提示信息
    print(f"🔥 启动 Pro_C4 训练流程（EfficientNet-B0）！")
    
    # 打印训练集和验证集样本数量
    print(f"📦 训练集: {train_size} | 验证集: {val_size}")

    # ==========================================
    # 步骤3: 训练循环 (核心训练流程 - EfficientNet-B0)
    # ==========================================
    # 外层循环: 遍历所有epoch(15轮完整数据集遍历)
    for epoch in range(EPOCHS):
        # ---------------- 阶段1: 训练模式 (前向传播+反向传播+参数更新) ----------------
        model.train()  # 切换到训练模式(启用Dropout随机失活、BatchNorm动量更新)
        train_loss_total = 0.0  # 初始化epoch总损失累加器
        # 创建可视化进度条,显示当前epoch进度和实时损失
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        # 内层循环: 遍历训练集的每个batch(64张图片为一组)
        for imgs, ages, genders in pbar:
            # 将数据从CPU内存移动到GPU显存(模型在GPU上,数据也必须在GPU)
            imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
            
            # 清零优化器梯度(不清零会累加上一次batch的梯度,导致更新错误)
            optimizer.zero_grad()
            
            # 混合精度前向传播(使用FP16半精度加速计算,节省显存)
            with autocast('cuda'):
                # 模型预测: 输入图片 → 输出年龄和性别预测
                # p_age形状: [batch_size, 1], 例如[64, 1]
                # p_gen形状: [batch_size, 2], 例如[64, 2] (男概率, 女概率)
                p_age, p_gen = model(imgs)
                
                # 计算年龄损失: MAE(平均绝对误差)
                # .squeeze()将[64,1]压缩为[64],与ages形状匹配
                loss_age = criterion_age(p_age.squeeze(), ages.float())
                
                # 计算性别损失: 交叉熵(带label_smoothing=0.1)
                loss_gen = criterion_gen(p_gen, genders)
                
                # 加权总损失: 年龄损失 + 性别损失×6
                # 为什么加权? 年龄损失(MAE)通常2-5,性别损失(交叉熵)通常0.1-0.5
                # 不加权会导致年龄损失主导,性别任务学不好
                loss = loss_age + (loss_gen * GEN_LOSS_WEIGHT)
            
            # 反向传播: 计算损失对每个模型参数的梯度
            # scaler.scale(): 将损失放大(如256倍),防止FP16精度丢失小梯度
            scaler.scale(loss).backward()
            
            # 反缩放梯度: 将放大的梯度还原到真实值(更新参数前必须还原)
            scaler.unscale_(optimizer)
            
            # 梯度裁剪: 限制梯度范数最大为1.0,防止梯度爆炸导致训练不稳定
            # 原理: 如果grad_norm > 1.0, 所有梯度等比例缩小
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新模型参数: 新参数 = 旧参数 - 学习率×梯度 - weight_decay×旧参数
            scaler.step(optimizer)
            
            # 动态调整缩放因子: 梯度溢出则缩小,正常则放大(提高FP16精度)
            scaler.update()
            
            # 更新学习率: 根据OneCycleLR策略调整(每个batch都调用)
            # 前20%步数warmup(学习率线性增长),后80%余弦衰减
            scheduler.step()

            # 累计本epoch的总损失(loss.item()是平均损失,需乘以batch_size得到总损失)
            train_loss_total += loss.item() * imgs.size(0)
            
            # 更新进度条显示: 在末尾显示当前batch的损失值(保留3位小数)
            # 显示效果: Epoch 1/15 [Train]: 62%|██████▏ | 97/156 [01:23<00:42, loss=4.123]
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # 计算本epoch的平均训练损失(总损失 / 训练集样本数)
        avg_train_loss = train_loss_total / train_size

        # ---------------- 阶段2: 验证模式 (评估模型性能,不更新参数) ----------------
        model.eval()  # 切换到评估模式(关闭Dropout随机失活,BatchNorm使用全局统计量)
        # 初始化验证指标累加器
        val_loss_total = 0.0       # 验证集总损失
        val_mae_total = 0.0        # 验证集年龄MAE总和
        val_acc_correct = 0        # 验证集性别预测正确数量
        
        # 禁用梯度计算(验证不需要反向传播,节省显存约50%,加速约30%)
        with torch.no_grad():
            # 遍历验证集的每个batch
            for imgs, ages, genders in val_loader:
                # 数据移动到GPU
                imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
                
                # 混合精度前向传播(验证也用FP16加速)
                with autocast('cuda'):
                    # 模型预测
                    p_age, p_gen = model(imgs)
                    
                    # 计算年龄损失(MAE)
                    loss_age = criterion_age(p_age.squeeze(), ages.float())
                    
                    # 计算性别损失(交叉熵)
                    loss_gen = criterion_gen(p_gen, genders)
                    
                    # 加权总损失(与训练一致)
                    loss = loss_age + (loss_gen * GEN_LOSS_WEIGHT)
                
                # 累计验证集总损失
                val_loss_total += loss.item() * imgs.size(0)
                
                # 累计年龄MAE(预测值与真实值的绝对差之和)
                # torch.abs(): 计算绝对值
                # .sum(): 求和整个batch的绝对误差
                # .item(): 转换为Python数值
                val_mae_total += torch.abs(p_age.squeeze() - ages.float()).sum().item()
                
                # 累计性别预测正确数量
                # torch.argmax(p_gen, 1): 取概率最大的类别索引(0=男,1=女)
                # == genders: 与真实标签比较,返回布尔数组[True, False, True, ...]
                # .sum(): 统计True的数量(即预测正确的样本数)
                val_acc_correct += (torch.argmax(p_gen, 1) == genders).sum().item()

        # 计算平均验证指标
        avg_val_loss = val_loss_total / val_size          # 平均验证损失
        avg_val_mae = val_mae_total / val_size            # 平均年龄误差(岁)
        avg_val_acc = val_acc_correct / val_size          # 性别准确率(0-1之间的小数)

        # 打印本epoch的验证结果
        # :.3f 表示保留3位小数, :.2f 表示保留2位小数, :.2% 表示百分比格式
        # 输出示例: 📈 验证结果 -> Loss: 3.456 | Age MAE: 4.23岁 | Gender Acc: 95.67%
        print(f"📈 验证结果 -> Loss: {avg_val_loss:.3f} | Age MAE: {avg_val_mae:.2f}岁 | Gender Acc: {avg_val_acc:.2%}")

        # ---------------- 阶段3: 保存模型权重和记录训练指标 ----------------
        # 将本epoch的训练和验证指标追加到历史记录列表
        history.append({
            "epoch": epoch + 1,                    # 当前epoch编号(从1开始)
            "train_loss": avg_train_loss,          # 平均训练损失
            "val_loss": avg_val_loss,              # 平均验证损失
            "val_mae": avg_val_mae,                # 平均年龄MAE(岁)
            "val_acc": avg_val_acc * 100           # 性别准确率(转换为百分比,如95.67)
        })
        
        # 保存模型权重到磁盘(每个epoch都保存,方便后续选择最佳模型)
        # state_dict()包含模型所有层的参数(权重和偏置)
        # 文件名示例: pro_eb0_epoch_1.pth, pro_eb0_epoch_2.pth, ...
        torch.save(model.state_dict(), f"{SAVE_DIR}/pro_eb0_epoch_{epoch+1}.pth")

    # ==========================================
    # 步骤4: 训练结束 - 生成研究报告图表和数据报告
    # ==========================================
    # 打印训练完成提示
    print("\n🎉 训练完成，正在生成研究报告 C4 图表与数据报告...")
    
    # 将训练历史列表转换为pandas DataFrame(表格数据结构)
    # DataFrame便于数据分析和保存为CSV文件
    df = pd.DataFrame(history)
    
    # 保存训练日志为CSV文件(可用Excel打开查看)
    # index=False表示不保存行索引
    df.to_csv(f"{SAVE_DIR}/plots/training_log_C4.csv", index=False)

    # ==========================================
    # 图表1: 训练损失和验证损失曲线(双线图)
    # ==========================================
    # 创建画布,设置尺寸为10×6英寸
    plt.figure(figsize=(10, 6))
    
    # 绘制训练损失曲线(蓝色实线,线宽2)
    plt.plot(df['epoch'], df['train_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2)
    
    # 绘制验证损失曲线(橙色虚线,线宽2)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Total Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # 设置图表标题,字体大小14
    plt.title('Training and Validation Loss over Epochs (C4)', fontsize=14)
    
    # 设置X轴标签,字体大小12
    plt.xlabel('Epoch', fontsize=12)
    
    # 设置Y轴标签,字体大小12
    plt.ylabel('Total Loss', fontsize=12)
    
    # 显示图例,字体大小12
    plt.legend(fontsize=12)
    
    # 添加网格线(虚线样式,透明度0.6)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存为300 DPI高分辨率PNG图片(适合研究报告使用)
    # bbox_inches='tight'自动裁剪空白边缘
    plt.savefig(f"{SAVE_DIR}/plots/loss_curve_C4.png", dpi=300, bbox_inches='tight')
    
    # 关闭当前图表,释放内存
    plt.close()

    # ==========================================
    # 图表2: 验证集指标曲线(双Y轴)
    # ==========================================
    # 创建画布和第一个Y轴(ax1),尺寸10×6英寸
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左Y轴: 性别准确率(绿色,越高越好)
    # 设置X轴标签
    ax1.set_xlabel('Epoch', fontsize=12)
    
    # 设置左Y轴标签,颜色为绿色(#2ca02c)
    ax1.set_ylabel('Gender Accuracy (%)', color='#2ca02c', fontsize=12)
    
    # 绘制性别准确率曲线(绿色实线,带圆形标记,标记大小4)
    ax1.plot(df['epoch'], df['val_acc'], color='#2ca02c', label='Gender Accuracy', linewidth=2, marker='o', markersize=4)
    
    # 设置左Y轴刻度标签颜色为绿色
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    
    # 右Y轴: 年龄MAE(红色,越低越好)
    # 创建共享X轴的第二个Y轴(ax2)
    ax2 = ax1.twinx()
    
    # 设置右Y轴标签,颜色为红色(#d62728)
    ax2.set_ylabel('Age MAE (Years)', color='#d62728', fontsize=12)
    
    # 绘制年龄MAE曲线(红色实线,带方形标记,标记大小4)
    ax2.plot(df['epoch'], df['val_mae'], color='#d62728', label='Age MAE', linewidth=2, marker='s', markersize=4)
    
    # 设置右Y轴刻度标签颜色为红色
    ax2.tick_params(axis='y', labelcolor='#d62728')

    # 设置图表标题
    plt.title('Validation Metrics: Gender Accuracy vs Age MAE (C4)', fontsize=14)
    
    # 自动调整布局防止标签重叠
    fig.tight_layout()
    
    # 添加网格线
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存为300 DPI高分辨率PNG图片
    plt.savefig(f"{SAVE_DIR}/plots/metrics_curve_C4.png", dpi=300, bbox_inches='tight')
    
    # 关闭图表,释放内存
    plt.close()

    # 打印保存完成提示,显示保存路径
    print(f"✅ 所有数据与高分辨率图表已保存至: {SAVE_DIR}/plots/")

if __name__ == "__main__":
    train()