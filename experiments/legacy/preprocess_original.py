import os
import cv2
import pandas as pd
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

# ==========================================
# 1. 核心配置
# ==========================================
SOURCE_DIR = "/root/my_age_project/data"
TARGET_DIR = "/root/my_age_project/data_aligned"
IMG_SIZE = 380
PADDING_RATIO = 0.35  # 增加到 35%，保留发际线和下巴特征

DIR_MAP = {
    "utk": "UTKFace",
    "afad": "AFAD-Full",
    "imdb_csv": "imdb-clean/imdb_train_new_1024.csv",
    "imdb_img": "imdb-clean/imdb-clean-1024/imdb-clean-1024"
}

# ==========================================
# 2. 对齐核心算法
# ==========================================
def align_face_v2(img, face):
    """基于眼睛关键点扶正人脸并外扩裁剪"""
    lmk = face.landmark_2d_106
    # 106点模型中：左眼中心索引约39，右眼中心索引约89
    left_eye = lmk[39]
    right_eye = lmk[89]
    
    # 计算旋转角度
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # 获取旋转矩阵并旋转原图
    center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    h, w = img.shape[:2]
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    # 裁剪范围计算
    x1, y1, x2, y2 = face.bbox.astype(int)
    bw, bh = x2 - x1, y2 - y1
    
    nx1 = max(0, int(x1 - bw * PADDING_RATIO))
    ny1 = max(0, int(y1 - bh * PADDING_RATIO))
    nx2 = min(w, int(x2 + bw * PADDING_RATIO))
    ny2 = min(h, int(y2 + bh * PADDING_RATIO))
    
    face_img = rotated_img[ny1:ny2, nx1:nx2]
    if face_img.size == 0: # 防止异常裁剪
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
    # 初始化模型 (仅初始化一次)
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    all_tasks = get_tasks()
    print(f"📊 扫描完成，共有 {len(all_tasks)} 个任务待处理...")

    for src, dst in tqdm(all_tasks, desc="对齐进度"):
        if os.path.exists(dst): continue
        
        img = cv2.imread(src)
        if img is None: continue
        
        faces = app.get(img)
        try:
            if len(faces) > 0:
                # 选置信度最高的人脸
                best_face = sorted(faces, key=lambda x: x.det_score)[-1]
                result = align_face_v2(img, best_face)
            else:
                # 保底：中心裁剪
                h, w = img.shape[:2]
                side = min(h, w)
                result = cv2.resize(img[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2], (IMG_SIZE, IMG_SIZE))
            
            cv2.imwrite(dst, result)
        except Exception:
            continue

    print(f"\n🎉 预处理全部完成！结果存储在: {TARGET_DIR}")