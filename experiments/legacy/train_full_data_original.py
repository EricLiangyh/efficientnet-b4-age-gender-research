import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# ==========================================
# 1. 配置：极致性能与泛化
# ==========================================
BASE_DIR = "/root/my_age_project"
DATA_DIR = f"{BASE_DIR}/data_aligned" 
SAVE_DIR = f"{BASE_DIR}/checkpoints_pro_v1"

BATCH_SIZE = 64         # 适配 RTX 4090/5090 显存
EPOCHS = 35             # 增强项较多，建议 35 轮以充分收敛
LR_MAX = 4e-4           # OneCycle 调度的最高学习率
GEN_LOSS_WEIGHT = 6.0   # 强化性别分支权重
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 模型定义 (EfficientNet-B4)
# ==========================================
class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练权重并添加 Dropout 防止过拟合
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Identity()
        )
        self.age_head = nn.Linear(in_feats, 1)    # 年龄回归
        self.gender_head = nn.Linear(in_feats, 2) # 性别分类

    def forward(self, x):
        feat = self.backbone(x)
        return self.age_head(feat), self.gender_head(feat)

# ==========================================
# 3. 数据集解析 (适配 Aligned 格式)
# ==========================================
class AlignedDataset(Dataset):
    """
    专门解析 preprocess_pro.py 生成的文件格式:
    格式: {age}_{g_code}_{source}_{original_name}
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 只扫描图片文件
        if not os.path.exists(root_dir):
            self.files = []
        else:
            self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        # 分隔文件名获取标签
        parts = file_name.split('_')
        
        try:
            age = float(parts[0])
            gender = int(parts[1]) # 0:男, 1:女
            
            img_path = os.path.join(self.root_dir, file_name)
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img, torch.tensor(age), torch.tensor(gender)
        except Exception as e:
            # 容错：遇到损坏文件随机跳转
            return self.__getitem__(torch.randint(0, len(self.files), (1,)).item())

# ==========================================
# 4. 工业级数据增强逻辑
# ==========================================
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((380, 380)),
        # 1. 几何变换
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        
        # 2. 颜色与光照模拟 (模拟不同环境/摄像头)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        
        # 3. 图像降级 (模拟模糊)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        
        # 4. 随机遮挡 (模拟口罩/障碍物)
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    ])

# ==========================================
# 5. 训练主循环
# ==========================================
def train():
    train_transform = get_train_transform()
    
    # 加载对齐后的子数据集
    subsets = ["UTKFace", "AFAD", "imdb"] 
    dataset_list = []
    for s in subsets:
        path = os.path.join(DATA_DIR, s)
        if os.path.exists(path):
            dataset_list.append(AlignedDataset(path, train_transform))
    
    if not dataset_list:
        print(f"❌ 错误：在 {DATA_DIR} 未找到任何对齐后的数据！请先运行 preprocess_pro.py")
        return

    full_dataset = ConcatDataset(dataset_list)
    # 高性能读取设置
    loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=16, pin_memory=True, prefetch_factor=3)

    model = AgeGenderModel().to(DEVICE)
    # 使用 AdamW 并增加权重衰减防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.05) 
    
    # OneCycle 调度器：先升温再降温，适合大规模训练
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, 
                                             steps_per_epoch=len(loader), epochs=EPOCHS,
                                             pct_start=0.2) 
    
    criterion_age = nn.L1Loss()
    criterion_gen = nn.CrossEntropyLoss(label_smoothing=0.1) # 标签平滑提升容错
    scaler = GradScaler('cuda')

    print(f"🔥 启动工业级 Pro 训练！总样本: {len(full_dataset)}")
    print(f"📢 增强项：透视、模糊、光照模拟、随机遮挡")

    for epoch in range(EPOCHS):
        model.train()
        running_mae, running_acc, total = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, ages, genders in pbar:
            imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
            
            optimizer.zero_grad()
            # 混合精度加速
            with autocast('cuda'):
                p_age, p_gen = model(imgs)
                loss_age = criterion_age(p_age.squeeze(), ages.float())
                loss_gen = criterion_gen(p_gen, genders)
                loss = loss_age + (loss_gen * GEN_LOSS_WEIGHT)
            
            scaler.scale(loss).backward()
            
            # 梯度裁剪：防止强增强导致梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 统计指标
            total += imgs.size(0)
            running_mae += torch.abs(p_age.squeeze() - ages.float()).sum().item()
            running_acc += (torch.argmax(p_gen, 1) == genders).sum().item()
            
            pbar.set_postfix(mae=f"{running_mae/total:.2f}y", acc=f"{running_acc/total:.1%}")

        # 每个 Epoch 保存一次
        save_path = f"{SAVE_DIR}/pro_eb4_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train()