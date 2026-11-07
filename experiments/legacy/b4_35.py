"""
年龄和性别预测模型训练脚本 (Pro_C3版本)
=========================================
模型架构: EfficientNet-B4 (双分支: 年龄回归 + 性别分类)
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
# 1. 核心配置参数
# ==========================================
# 项目根目录(所有路径基于此)
BASE_DIR = "/root/my_age_project"
# 对齐后的人脸数据目录(由preprocess_pro.py生成)
DATA_DIR = f"{BASE_DIR}/data_aligned" 
# 模型权重和训练日志保存目录
SAVE_DIR = f"{BASE_DIR}/checkpoints_pro_v1"

# 训练超参数
BATCH_SIZE = 64           # 批次大小: 每次训练64张图片(显存和速度的平衡)
EPOCHS = 35               # 训练轮数: 完整遍历数据集35次
LR_MAX = 4e-4             # 最大学习率: 0.0004(OneCycleLR的峰值学习率)
GEN_LOSS_WEIGHT = 6.0     # 性别损失权重: 性别损失×6(平衡年龄和性别任务的重要性)

# 设备选择: 优先使用GPU,否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存目录(如果不存在)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/plots", exist_ok=True)  # 专门存放训练曲线图的子目录

# ==========================================
# 2. 模型定义: EfficientNet-B4双分支架构
# ==========================================
class AgeGenderModel(nn.Module):
    """
    年龄和性别预测模型
    
    架构设计:
    - 主干网络: EfficientNet-B4 (ImageNet预训练权重)
    - 年龄分支: 回归任务 (输出1个连续值)
    - 性别分支: 分类任务 (输出2个类别概率)
    
    特征提取流程:
    输入(380×380×3) → EfficientNet-B4 → 1792维特征向量
                                        ↓
                              ┌─────────┴─────────┐
                              ↓                   ↓
                         年龄预测层            性别预测层
                         (Linear)             (Linear)
                              ↓                   ↓
                          1个值               2个概率
    """
    def __init__(self):
        super().__init__()
        # 加载EfficientNet-B4预训练模型(ImageNet权重)
        # 预训练权重已学习通用视觉特征(边缘、纹理、形状等)
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        
        # 获取原始分类层的输入特征维度(EfficientNet-B4为1792)
        in_feats = self.backbone.classifier[1].in_features
        
        # 替换原始分类层: 移除ImageNet的1000类分类头
        # 保留Dropout层(防止过拟合),但移除最后的Linear层
        # nn.Identity()是恒等映射,直接输出特征向量
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),  # 30%的神经元随机失活
            nn.Identity()                      # 直接输出1792维特征
        )
        
        # 年龄预测头: 回归任务(输出连续值)
        # 输入: 1792维特征 → 输出: 1个年龄值
        self.age_head = nn.Linear(in_feats, 1)    
        
        # 性别预测头: 分类任务(输出二分类概率)
        # 输入: 1792维特征 → 输出: 2个类别(男/女)的logits
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
        feat = self.backbone(x)  # feat形状: [batch_size, 1792]
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
    
    参数:
        root_dir: 数据集根目录
        transform: 图像变换(数据增强)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 检查目录是否存在
        if not os.path.exists(root_dir):
            self.files = []
        else:
            # 扫描所有图像文件(支持jpg/png/jpeg格式)
            self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self): 
        """返回数据集大小"""
        return len(self.files)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            (image, age_label, gender_label): 
                - image: 图像张量
                - age_label: 年龄标签(浮点数)
                - gender_label: 性别标签(0=男,1=女)
        """
        file_name = self.files[idx]
        # 解析文件名: 按'_'分割
        parts = file_name.split('_')
        try:
            # 提取年龄和性别标签
            age = float(parts[0])           # 第一部分: 年龄(如25.0)
            gender = int(parts[1])          # 第二部分: 性别(0或1)
            
            # 加载图像
            img_path = os.path.join(self.root_dir, file_name)
            img = Image.open(img_path).convert('RGB')  # 确保为RGB三通道
            
            # 应用数据增强(如果有)
            if self.transform: 
                img = self.transform(img)
            
            return img, torch.tensor(age), torch.tensor(gender)
        except Exception:
            # 异常处理: 如果文件损坏或格式错误,随机选择另一个样本
            # 这保证了训练不会因个别坏文件而中断
            return self.__getitem__(torch.randint(0, len(self.files), (1,)).item())

# ==========================================
# 4. 数据增强策略
# ==========================================
def get_train_transform():
    """
    训练集数据增强流水线
    
    目的: 通过随机变换扩充数据多样性,提升模型泛化能力
    
    增强策略(按执行顺序):
    1. Resize: 统一缩放到380×380
    2. RandomHorizontalFlip: 50%概率水平翻转(模拟左右脸)
    3. RandomRotation: ±15度随机旋转(模拟头部姿态变化)
    4. RandomPerspective: 50%概率透视变换(模拟拍摄角度)
    5. ColorJitter: 随机调整亮度/对比度/饱和度/色调(模拟光照变化)
    6. RandomGrayscale: 10%概率转灰度(降低颜色依赖)
    7. GaussianBlur: 20%概率高斯模糊(模拟对焦不准)
    8. ToTensor: 转换为PyTorch张量并归一化到[0,1]
    9. Normalize: ImageNet标准化(匹配预训练权重分布)
    10. RandomErasing: 30%概率随机遮挡(提升鲁棒性)
    
    返回:
        transforms.Compose对象
    """
    return transforms.Compose([
        transforms.Resize((380, 380)),  # 统一输入尺寸
        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转: 左右脸对称
        transforms.RandomRotation(degrees=15),   # 随机旋转±15度
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 透视变换
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 颜色抖动
        transforms.RandomGrayscale(p=0.1),  # 灰度化
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),  # 模糊
        transforms.ToTensor(),  # 转张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet标准化
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),  # 随机遮挡
    ])

def get_val_transform():
    """
    验证集数据变换(无随机增强)
    
    与训练集的区别:
    - 不包含任何随机变换(翻转/旋转/颜色抖动等)
    - 只保留必要的Resize和Normalize
    - 保证评估结果的稳定性和可重复性
    
    返回:
        transforms.Compose对象
    """
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
    - 这个包装器允许在数据加载时动态应用变换
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
    主训练函数
    
    训练流程:
    1. 加载三个数据集(UTKFace, AFAD, IMDB)
    2. 划分80%训练集和20%验证集
    3. 初始化模型、优化器、学习率调度器
    4. 循环训练EPOCHS轮
    5. 每轮保存模型权重和训练指标
    6. 训练结束后生成可视化图表
    """
    # ==========================================
    # 步骤1: 数据准备
    # ==========================================
    # 指定要加载的数据集子集
    subsets = ["UTKFace", "AFAD", "imdb"] 
    dataset_list = []
    
    # 遍历每个数据集目录
    for s in subsets:
        path = os.path.join(DATA_DIR, s)
        if os.path.exists(path):
            # 加载数据集(不应用transform,后续动态应用)
            dataset_list.append(AlignedDataset(path, transform=None))
            
    # 检查是否找到数据
    if not dataset_list:
        print(f"❌ 错误：在 {DATA_DIR} 未找到任何对齐后的数据！")
        return

    # 合并多个数据集为一个大集合
    full_dataset = ConcatDataset(dataset_list)
    
    # 划分训练集和验证集(80/20比例)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    # 使用固定随机种子保证可重复性
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # 使用TransformSubset包装,分别应用不同的Transform
    train_dataset = TransformSubset(train_subset, get_train_transform())  # 训练集: 有数据增强
    val_dataset = TransformSubset(val_subset, get_val_transform())        # 验证集: 无数据增强

    # 创建DataLoader(用于批量加载数据)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # ==========================================
    # 步骤2: 模型与优化器初始化
    # ==========================================
    # 创建模型并移动到GPU
    model = AgeGenderModel().to(DEVICE)
    
    # AdamW优化器(带权重衰减的正则化)
    # weight_decay=0.05的作用:
    #   损失函数 = 原始损失 + 0.05 × 所有参数的平方和
    #   - 惩罚过大的权重值
    #   - 防止模型"死记硬背"训练数据
    #   - 提升泛化能力
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.05) 
    
    # OneCycleLR学习率调度器
    # 学习率先线性增长到LR_MAX,再余弦衰减到接近0
    # pct_start=0.2表示前20%的步数用于warmup
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, 
                                             steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2) 
    
    # 损失函数定义
    criterion_age = nn.L1Loss()  # 年龄损失: MAE(平均绝对误差)
    
    # 性别损失: 交叉熵(带标签平滑)
    # label_smoothing=0.1的作用:
    #   不标签平滑时:
    #     真实标签: [1, 0]  (100%男性, 0%女性)
    #   标签平滑后:
    #     真实标签: [0.9, 0.1]  (90%男性, 10%女性)
    #   为什么要标签平滑?
    #     1. 防止模型过度自信
    #        - 没有平滑: 模型会拼命让预测接近[1, 0],容易过拟合
    #        - 有平滑: 模型学到"90%把握是男性"就够了
    #     2. 提升泛化能力
    #        - 真实世界的数据有噪声(比如标注错误)
    #        - 标签平滑让模型对噪声更鲁棒
    criterion_gen = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 混合精度训练: 使用FP16加速训练并减少显存占用
    scaler = GradScaler('cuda')

    # 训练历史记录器
    history = []

    print(f"🔥 启动 Pro_C3 训练流程！")
    print(f"📦 训练集: {train_size} | 验证集: {val_size}")

    # ==========================================
    # 步骤3: 训练循环 (核心训练流程)
    # ==========================================
    # 外层循环: 遍历所有epoch(35轮完整数据集遍历)
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
            # 显示效果: Epoch 1/35 [Train]: 62%|██████▏ | 97/156 [01:23<00:42, loss=4.123]
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
        # 文件名示例: pro_eb4_epoch_1.pth, pro_eb4_epoch_2.pth, ...
        torch.save(model.state_dict(), f"{SAVE_DIR}/pro_eb4_epoch_{epoch+1}.pth")

    # ==========================================
    # 步骤4: 训练结束 - 生成研究报告图表和数据报告
    # ==========================================
    print("\n🎉 训练完成，正在生成研究报告 C3 图表与数据报告...")
    
    # 将训练历史转换为DataFrame并保存为CSV
    df = pd.DataFrame(history)
    df.to_csv(f"{SAVE_DIR}/plots/training_log_C3.csv", index=False)

    # ==========================================
    # 图表1: 训练损失和验证损失曲线(双线图)
    # ==========================================
    plt.figure(figsize=(10, 6))
    # 绘制训练损失曲线(蓝色实线)
    plt.plot(df['epoch'], df['train_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2)
    # 绘制验证损失曲线(橙色虚线)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Total Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Training and Validation Loss over Epochs (C3)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    # 保存为300 DPI高分辨率图片(适合研究报告使用)
    plt.savefig(f"{SAVE_DIR}/plots/loss_curve_C3.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 图表2: 验证集指标曲线(双Y轴)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左Y轴: 性别准确率(绿色,越高越好)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Gender Accuracy (%)', color='#2ca02c', fontsize=12)
    ax1.plot(df['epoch'], df['val_acc'], color='#2ca02c', label='Gender Accuracy', linewidth=2, marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    
    # 右Y轴: 年龄MAE(红色,越低越好)
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
    ax2.set_ylabel('Age MAE (Years)', color='#d62728', fontsize=12)
    ax2.plot(df['epoch'], df['val_mae'], color='#d62728', label='Age MAE', linewidth=2, marker='s', markersize=4)
    ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.title('Validation Metrics: Gender Accuracy vs Age MAE (C3)', fontsize=14)
    fig.tight_layout()  # 自动调整布局防止标签重叠
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/plots/metrics_curve_C3.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 所有数据与高分辨率图表已保存至: {SAVE_DIR}/plots/")

if __name__ == "__main__":
    train()