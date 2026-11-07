import os
import cv2
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

# ==========================================
# 1. 核心配置 (C12: 未对齐对照)
# ==========================================
BASE_DIR = "/root/my_age_project"
SOURCE_DIR = f"{BASE_DIR}/data"
TARGET_DIR = f"{BASE_DIR}/C12/data_not_aligned"
IMG_SIZE = 380
PADDING_RATIO = 0.0  # C12: 取消扩展，紧贴检测框裁剪

DIR_MAP = {
    "utk": "UTKFace",
    "afad": "AFAD-Full",
    "imdb_csv": "imdb-clean/imdb_train_new_1024.csv",
    "imdb_img": "imdb-clean/imdb-clean-1024/imdb-clean-1024"
}

# ==========================================
# 2. 未对齐处理 (C12对照组)
# ==========================================
def crop_face_no_align(img, face):
    """C12对照组：不旋转，直接用检测框裁剪（可选小幅扩展）"""
    # 直接使用检测框，不做旋转对齐
    x1, y1, x2, y2 = face.bbox.astype(int)
    bw, bh = x2 - x1, y2 - y1
    
    # 紧贴检测框裁剪（PADDING_RATIO=0）
    nx1 = max(0, int(x1 - bw * PADDING_RATIO))
    ny1 = max(0, int(y1 - bh * PADDING_RATIO))
    nx2 = min(img.shape[1], int(x2 + bw * PADDING_RATIO))
    ny2 = min(img.shape[0], int(y2 + bh * PADDING_RATIO))
    
    face_img = img[ny1:ny2, nx1:nx2]
    if face_img.size == 0:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

# ==========================================
# 3. 数据集任务扫描逻辑
# ==========================================
def get_tasks():
    tasks = []
    
    # 1. UTKFace
    utk_path = os.path.join(SOURCE_DIR, DIR_MAP["utk"])
    if os.path.exists(utk_path):
        out_dir = os.path.join(TARGET_DIR, "UTKFace")
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(utk_path):
            if f.lower().endswith(('.jpg', '.png')):
                tasks.append((os.path.join(utk_path, f), os.path.join(out_dir, f)))

    # 2. AFAD
    afad_path = os.path.join(SOURCE_DIR, DIR_MAP["afad"])
    if os.path.exists(afad_path):
        out_dir = os.path.join(TARGET_DIR, "AFAD")
        os.makedirs(out_dir, exist_ok=True)
        for age in [d for d in os.listdir(afad_path) if d.isdigit()]:
            for g_f in ['111', '112']:
                gp = os.path.join(afad_path, age, g_f)
                if os.path.exists(gp):
                    g_code = "0" if g_f == '111' else "1"
                    for f in os.listdir(gp):
                        save_name = f"{age}_{g_code}_afad_{f}"
                        tasks.append((os.path.join(gp, f), os.path.join(out_dir, save_name)))

    # 3. IMDB
    csv_p = os.path.join(SOURCE_DIR, DIR_MAP["imdb_csv"])
    img_r = os.path.join(SOURCE_DIR, DIR_MAP["imdb_img"])
    if os.path.exists(csv_p):
        out_dir = os.path.join(TARGET_DIR, "imdb")
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(csv_p)
        for _, row in df.iterrows():
            g_raw = str(row['gender']).upper()
            g_code = "1" if g_raw in ['F', 'FEMALE', '1.0', '1'] else "0"
            save_name = f"{int(row['age'])}_{g_code}_imdb_{os.path.basename(row['filename'])}"
            tasks.append((os.path.join(img_r, row['filename']), os.path.join(out_dir, save_name)))
            
    return tasks

# ==========================================
# 4. 执行入口
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("C12：人脸对齐消融实验 - 未对齐数据预处理")
    print("="*60)
    print(f"源数据目录: {SOURCE_DIR}")
    print(f"输出目录: {TARGET_DIR}")
    print(f"裁剪策略: 紧贴检测框 (PADDING_RATIO=0)")
    print(f"旋转对齐: 禁用\n")
    
    # 初始化模型
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    all_tasks = get_tasks()
    print(f"📊 扫描完成，共有 {len(all_tasks)} 个任务待处理...")

    for src, dst in tqdm(all_tasks, desc="未对齐处理进度"):
        if os.path.exists(dst): continue
        
        img = cv2.imread(src)
        if img is None: continue
        
        faces = app.get(img)
        try:
            if len(faces) > 0:
                best_face = sorted(faces, key=lambda x: x.det_score)[-1]
                result = crop_face_no_align(img, best_face)
            else:
                h, w = img.shape[:2]
                side = min(h, w)
                result = cv2.resize(img[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2], (IMG_SIZE, IMG_SIZE))
            
            cv2.imwrite(dst, result)
        except Exception:
            continue

    print(f"\n🎉 未对齐数据预处理完成！")
    print(f"结果存储在: {TARGET_DIR}")
