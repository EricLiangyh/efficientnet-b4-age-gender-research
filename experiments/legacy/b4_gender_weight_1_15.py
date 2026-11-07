"""
年龄和性别预测模型训练脚本 (C5实验 - 损失权重λ=1.0对照)
==========================================================
实验目的: 验证性别损失权重对模型性能的影响

与C3(GEN_LOSS_WEIGHT=6.0)的区别:
  - GEN_LOSS_WEIGHT = 1.0 (性别损失不放大)
  - 其他配置完全相同(EfficientNet-B4, 数据增强, 学习率等)
  
假设: λ=6.0时性别分类更准确,但可能影响年龄回归精度
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
import pandas as pd          
import matplotlib.pyplot as plt 

# ==========================================
# 1. 核心配置参数 (C5实验 - 损失权重λ=1.0)
# ==========================================
# 项目根目录(所有路径基于此)
BASE_DIR = "/root/my_age_project"
# 对齐后的人脸数据目录(由preprocess_pro.py生成)
DATA_DIR = f"{BASE_DIR}/data_aligned" 
# 模型权重和训练日志保存目录(λ=1.0实验独立保存)
SAVE_DIR = f"{BASE_DIR}/checkpoints_pro_v1_lambda1"

# 训练超参数
BATCH_SIZE = 64           # 批次大小: 每次训练64张图片
EPOCHS = 15               # 训练轮数: 15轮(对照实验)
LR_MAX = 4e-4             # 最大学习率: 0.0004
GEN_LOSS_WEIGHT = 1.0     # 性别损失权重: 1.0(不放大,与C3的6.0对比)

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
    年龄和性别预测模型 - C5实验使用EfficientNet-B4
    
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
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        # 获取原始分类层的输入特征维度(EfficientNet-B4为1792)
        in_feats = self.backbone.classifier[1].in_features
        # 替换原始分类层: 移除ImageNet的1000类分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),  # 30%的神经元随机失活
            nn.Identity()                      # 直接输出1792维特征
        )
        # 年龄预测头: 回归任务(输出连续值)
        self.age_head = nn.Linear(in_feats, 1)    
        # 性别预测头: 分类任务(输出二分类概率)
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
    主训练函数 - C5实验(损失权重λ=1.0)
    
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
    subsets = ["UTKFace", "AFAD", "imdb"] 
    dataset_list = []
    for s in subsets:
        path = os.path.join(DATA_DIR, s)
        if os.path.exists(path):
            dataset_list.append(AlignedDataset(path, transform=None))
            
    if not dataset_list:
        print(f"❌ 错误：在 {DATA_DIR} 未找到任何对齐后的数据！")
        return

    full_dataset = ConcatDataset(dataset_list)
    
    # 划分 80% 训练集，20% 验证集
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # 使用 TransformSubset 包装，分别应用不同的 Transform
    train_dataset = TransformSubset(train_subset, get_train_transform())
    val_dataset = TransformSubset(val_subset, get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # ==========================================
    # 步骤2: 模型与优化器初始化
    # ==========================================
    model = AgeGenderModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.05) 
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, 
                                             steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2) 
    criterion_age = nn.L1Loss()
    criterion_gen = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')

    history = []

    print(f"🔥 启动 C5 训练流程（策略 C1：λ=1.0）！")
    print(f"📦 训练集: {train_size} | 验证集: {val_size}")

    # ==========================================
    # 步骤3: 训练循环
    # ==========================================
    for epoch in range(EPOCHS):
        # ---------------- 阶段1: 训练 ----------------
        model.train()
        train_loss_total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for imgs, ages, genders in pbar:
            imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda'):
                p_age, p_gen = model(imgs)
                loss_age = criterion_age(p_age.squeeze(), ages.float())
                loss_gen = criterion_gen(p_gen, genders)
                loss = loss_age + (loss_gen * GEN_LOSS_WEIGHT)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_total += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        avg_train_loss = train_loss_total / train_size

        # ---------------- 阶段2: 验证 ----------------
        model.eval()
        val_loss_total, val_mae_total, val_acc_correct = 0.0, 0.0, 0
        
        with torch.no_grad():
            for imgs, ages, genders in val_loader:
                imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
                with autocast('cuda'):
                    p_age, p_gen = model(imgs)
                    loss_age = criterion_age(p_age.squeeze(), ages.float())
                    loss_gen = criterion_gen(p_gen, genders)
                    loss = loss_age + (loss_gen * GEN_LOSS_WEIGHT)
                
                val_loss_total += loss.item() * imgs.size(0)
                val_mae_total += torch.abs(p_age.squeeze() - ages.float()).sum().item()
                val_acc_correct += (torch.argmax(p_gen, 1) == genders).sum().item()

        avg_val_loss = val_loss_total / val_size
        avg_val_mae = val_mae_total / val_size
        avg_val_acc = val_acc_correct / val_size

        print(f"📈 验证结果 -> Loss: {avg_val_loss:.3f} | Age MAE: {avg_val_mae:.2f}岁 | Gender Acc: {avg_val_acc:.2%}")

        # ---------------- 阶段3: 保存模型和记录 ----------------
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mae": avg_val_mae,
            "val_acc": avg_val_acc * 100
        })
        torch.save(model.state_dict(), f"{SAVE_DIR}/pro_eb4_epoch_{epoch+1}.pth")

    # ==========================================
    # 步骤4: 训练结束 - 生成研究报告图表和数据报告
    # ==========================================
    print("\n🎉 训练完成，正在生成研究报告 C5 图表与数据报告...")
    df = pd.DataFrame(history)
    df.to_csv(f"{SAVE_DIR}/plots/training_log_C5_lambda1.csv", index=False)

    # ==========================================
    # 图表1: 训练损失和验证损失曲线
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Total Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Training and Validation Loss over Epochs (C5 - λ=1.0)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/plots/loss_curve_C5_lambda1.png", dpi=300, bbox_inches='tight')
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
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Age MAE (Years)', color='#d62728', fontsize=12)
    ax2.plot(df['epoch'], df['val_mae'], color='#d62728', label='Age MAE', linewidth=2, marker='s', markersize=4)
    ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.title('Validation Metrics: Gender Accuracy vs Age MAE (C5 - λ=1.0)', fontsize=14)
    fig.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/plots/metrics_curve_C5_lambda1.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 所有数据与高分辨率图表已保存至: {SAVE_DIR}/plots/")

if __name__ == "__main__":
    train()
