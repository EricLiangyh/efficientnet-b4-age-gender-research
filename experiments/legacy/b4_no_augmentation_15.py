"""
年龄和性别预测模型训练脚本 (C5实验 - 无数据增强对照)
==========================================================
实验目的: 验证数据增强对模型泛化能力的影响

与C3(标准数据增强)的区别:
  - 训练集仅使用Resize + Normalize,无任何随机增强
  - 移除了: 水平翻转、旋转、透视变换、颜色抖动、灰度化、高斯模糊、随机擦除
  - 其他配置完全相同(EfficientNet-B4, 15 epochs, GEN_LOSS_WEIGHT=6.0)

假设: 数据增强能提升模型在未见数据上的泛化能力,减少过拟合
"""
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

# ==================== 实验配置参数 ====================
BASE_DIR = "/root/my_age_project"
DATA_DIR = f"{BASE_DIR}/data_aligned" 
SAVE_DIR = f"{BASE_DIR}/checkpoints_pro_v1_no_aug"

BATCH_SIZE = 64         
EPOCHS = 15             
LR_MAX = 4e-4           
GEN_LOSS_WEIGHT = 6.0   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/plots", exist_ok=True) 

# ==================== 模型定义 ====================
class AgeGenderModel(nn.Module):
    """
    年龄和性别预测模型 (EfficientNet-B4主干)
    
    架构设计:
    - 主干网络: EfficientNet-B4 (ImageNet预训练权重)
    - 年龄分支: 回归任务 (输出1个连续值)
    - 性别分支: 分类任务 (输出2个类别概率)
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Identity()
        )
        self.age_head = nn.Linear(in_feats, 1)    
        self.gender_head = nn.Linear(in_feats, 2) 

    def forward(self, x):
        feat = self.backbone(x)
        return self.age_head(feat), self.gender_head(feat)

# ==================== 数据集定义 ====================
class AlignedDataset(Dataset):
    """
    对齐人脸数据集
    
    文件名格式: {age}_{gender}_{...}.jpg
    例如: 25_0_12345.jpg (年龄25岁,性别0=女性)
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

# ==================== 数据变换定义 (C5: 无增强) ====================
def get_train_transform():
    """
    训练集数据变换 - 仅基础处理,无随机增强
    
    与C3的区别: 移除了所有随机数据增强操作
    - 无水平翻转 (RandomHorizontalFlip)
    - 无旋转 (RandomRotation)
    - 无透视变换 (RandomPerspective)
    - 无颜色抖动 (ColorJitter)
    - 无灰度化 (RandomGrayscale)
    - 无高斯模糊 (GaussianBlur)
    - 无随机擦除 (RandomErasing)
    """
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_val_transform(): 
    """验证集数据变换 - 仅基础处理"""
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ==================== 训练主循环 ====================
class TransformSubset(Dataset):
    """为数据集应用变换的包装类"""
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
    C5实验训练流程 - 无数据增强对照
    
    训练策略:
    1. 加载三个数据集(UTKFace, AFAD, imdb)
    2. 80/20划分训练集/验证集
    3. 使用AdamW优化器 + OneCycleLR学习率调度
    4. 混合精度训练(AMP)加速
    5. 梯度裁剪防止梯度爆炸
    """
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
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_dataset = TransformSubset(train_subset, get_train_transform())
    val_dataset = TransformSubset(val_subset, get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = AgeGenderModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.05) 
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, 
                                             steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2) 
    criterion_age = nn.L1Loss()
    criterion_gen = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')

    history = []

    print(f"🔥 启动 C5 训练流程（策略 A：无数据增强）！")
    print(f"📦 训练集: {train_size} | 验证集: {val_size}")

    for epoch in range(EPOCHS):
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

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mae": avg_val_mae,
            "val_acc": avg_val_acc * 100
        })
        torch.save(model.state_dict(), f"{SAVE_DIR}/pro_eb4_epoch_{epoch+1}.pth")

    print("\n🎉 训练完成，正在生成研究报告 C5 图表与数据报告...")
    df = pd.DataFrame(history)
    df.to_csv(f"{SAVE_DIR}/plots/training_log_C5_no_aug.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Total Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Training and Validation Loss over Epochs (C5 - No Augmentation)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/plots/loss_curve_C5_no_aug.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Gender Accuracy (%)', color='#2ca02c', fontsize=12)
    ax1.plot(df['epoch'], df['val_acc'], color='#2ca02c', label='Gender Accuracy', linewidth=2, marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Age MAE (Years)', color='#d62728', fontsize=12)
    ax2.plot(df['epoch'], df['val_mae'], color='#d62728', label='Age MAE', linewidth=2, marker='s', markersize=4)
    ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.title('Validation Metrics: Gender Accuracy vs Age MAE (C5 - No Augmentation)', fontsize=14)
    fig.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/plots/metrics_curve_C5_no_aug.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 所有数据与高分辨率图表已保存至: {SAVE_DIR}/plots/")

if __name__ == "__main__":
    train()
