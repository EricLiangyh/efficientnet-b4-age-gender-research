"""
年龄和性别预测模型训练脚本 (C9实验 - ResNet-50骨干网络对照)
==========================================================
实验目的: 验证不同骨干网络架构对模型性能的影响

与C3(EfficientNet-B4)的区别:
  - 骨干网络: ResNet-50 (残差网络,50层)
  - 特征维度: 2048维 (vs EfficientNet-B4的1792维)
  - 架构特点: 残差连接解决深层网络退化问题
  - 其他配置完全相同(15 epochs, GEN_LOSS_WEIGHT=6.0, 数据增强)

ResNet-50特点:
  - 使用残差块(Residual Block)实现跳跃连接
  -  Bottleneck结构: 1×1 → 3×3 → 1×1 卷积
  -  参数量约25.6M,计算量约4.1B FLOPs
  -  在图像分类任务上表现稳定,是经典基线模型
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

# ==========================================
# 1. 配置：C9 ResNet50 对照实验 (15 epochs)
# ==========================================
BASE_DIR = "/root/my_age_project"
DATA_DIR = f"{BASE_DIR}/data_aligned"
SAVE_DIR = f"{BASE_DIR}/C9/checkpoints_resnet50_15"

BATCH_SIZE = 64         
EPOCHS = 15             
LR_MAX = 4e-4           
GEN_LOSS_WEIGHT = 6.0   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(f"{SAVE_DIR}/plots", exist_ok=True)

# ==========================================
# 2. 模型定义 (ResNet-50)
# ==========================================
class AgeGenderModel(nn.Module):
    """
    年龄和性别预测模型 (ResNet-50主干)
    
    架构设计:
    - 主干网络: ResNet-50 (ImageNet预训练权重)
    - 年龄分支: 回归任务 (输出1个连续值)
    - 性别分支: 分类任务 (输出2个类别概率)
    
    特征提取流程:
    输入(380×380×3) → ResNet-50 → 2048维特征向量
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
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.age_head = nn.Linear(in_feats, 1)    
        self.gender_head = nn.Linear(in_feats, 2) 

    def forward(self, x):
        feat = self.backbone(x)
        return self.age_head(feat), self.gender_head(feat)

# ==========================================
# 3. 数据集解析 
# ==========================================
class AlignedDataset(Dataset):
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
# 4. 数据增强逻辑
# ==========================================
def get_train_transform():
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
    return transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ==========================================
# 5. 训练与自动化记录主循环
# ==========================================
class TransformSubset(Dataset):
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
    print("="*60)
    print("C9：ResNet-50 骨干网络对照实验")
    print("="*60)
    print(f"数据目录: {DATA_DIR}")
    print(f"保存目录: {SAVE_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"设备: {DEVICE}\n")
    
    # --- 数据准备 ---
    subsets = ["UTKFace", "AFAD", "imdb"] 
    dataset_list = []
    for s in subsets:
        path = os.path.join(DATA_DIR, s)
        if os.path.exists(path):
            dataset_list.append(AlignedDataset(path, transform=None))
            
    if not dataset_list:
        print(f"❌ 错误：在 {DATA_DIR} 未找到任何数据！")
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
    print(f"🔥 启动 ResNet-50 训练流程！")
    print(f"📦 训练集: {train_size} | 验证集: {val_size}\n")

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

        print(f"📈 验证结果 -> Loss: {avg_val_loss:.3f} | Age MAE: {avg_val_mae:.2f}岁 | Gender Acc: {avg_val_acc:.2%}\n")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mae": avg_val_mae,
            "val_acc": avg_val_acc * 100
        })
        torch.save(model.state_dict(), f"{SAVE_DIR}/resnet50_epoch_{epoch+1}.pth")

    print("\n🎉 ResNet-50训练完成，正在生成研究报告图表与数据报告...")
    df = pd.DataFrame(history)
    df.to_csv(f"{SAVE_DIR}/plots/training_log_resnet50_15.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Total Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.title('Training and Validation Loss over Epochs (ResNet50 - 15 Epochs)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/plots/loss_curve_resnet50_15.png", dpi=300, bbox_inches='tight')
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

    plt.title('Validation Metrics: Gender Accuracy vs Age MAE (ResNet50 - 15 Epochs)', fontsize=14)
    fig.tight_layout()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"{SAVE_DIR}/plots/metrics_curve_resnet50_15.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 所有数据与图表已保存至: {SAVE_DIR}/plots/")

if __name__ == "__main__":
    train()
