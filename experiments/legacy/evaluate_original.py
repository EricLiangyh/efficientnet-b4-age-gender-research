import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
# 修正：sklearn.metrics 没有 rmse_score，直接使用内置计算
from sklearn.metrics import classification_report, confusion_matrix 

# ==========================================
# 1. 配置评估环境
# ==========================================
BASE_DIR = "/root/my_age_project"
DATA_DIR = f"{BASE_DIR}/data_aligned" 
MODEL_PATH = f"{BASE_DIR}/checkpoints_pro_v1/pro_eb4_epoch_35.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# ==========================================
# 2. 模型定义 (对齐 train_pro.py)
# ==========================================
class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 必须与 train_pro.py 结构完全相同
        self.backbone = models.efficientnet_b4(weights=None)
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Identity()
        )
        self.age_head = nn.Linear(in_feats, 1)    # 变量名必须是 age_head
        self.gender_head = nn.Linear(in_feats, 2) # 变量名必须是 gender_head

    def forward(self, x):
        feat = self.backbone(x)
        return self.age_head(feat), self.gender_head(feat)

# ==========================================
# 3. 数据集加载 (对齐 preprocess_pro.py 目录名)
# ==========================================
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        # 对齐 preprocess_pro.py 中的目录名: UTKFace, AFAD, imdb
        for subset in ["UTKFace", "AFAD", "imdb"]:
            sub_path = os.path.join(root_dir, subset)
            if not os.path.exists(sub_path): continue
            for f in os.listdir(sub_path):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append(os.path.join(sub_path, f))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        # 针对 UTKFace 和 AFAD/IMDB 统一解析逻辑
        try:
            age = float(parts[0])
            gender = int(parts[1])
            if self.transform: img = self.transform(img)
            return img, age, gender
        except:
            return self.__getitem__(0)

def get_test_loader():
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    full_dataset = FaceDataset(DATA_DIR, transform=val_transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # 强制不使用随机种子，或使用特定种子，确保与训练集划分逻辑一致
    _, test_dataset = random_split(full_dataset, [train_size, test_size], 
                                  generator=torch.Generator().manual_seed(42))
    return DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 4. 执行评估 (计算 MAE, RMSE, Acc)
# ==========================================
def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到权重文件: {MODEL_PATH}")
        return

    model = AgeGenderModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_loader = get_test_loader()
    true_ages, pred_ages = [], []
    true_gens, pred_gens = [], []

    print(f"🚀 正在测试集上验证 {len(test_loader.dataset)} 个样本...")
    with torch.no_grad():
        for imgs, ages, genders in tqdm(test_loader):
            imgs = imgs.to(DEVICE)
            p_age, p_gen = model(imgs)
            
            true_ages.extend(ages.numpy())
            pred_ages.extend(p_age.cpu().squeeze().numpy())
            true_gens.extend(genders.numpy())
            pred_gens.extend(torch.argmax(p_gen, 1).cpu().numpy())

    # 指标计算
    true_ages, pred_ages = np.array(true_ages), np.array(pred_ages)
    mae = np.mean(np.abs(true_ages - pred_ages))
    rmse = np.sqrt(np.mean((true_ages - pred_ages)**2))
    
    print("\n📊 评估报告")
    print(f"年龄 MAE: {mae:.2f} | 年龄 RMSE: {rmse:.2f}")
    print(f"\n性别分类报告:\n", classification_report(true_gens, pred_gens, target_names=['Male', 'Female']))

if __name__ == "__main__":
    evaluate()