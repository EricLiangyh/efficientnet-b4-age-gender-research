import os
import numpy as np
from collections import Counter

BASE_DIR = "/root/my_age_project"
DATA_DIR = f"{BASE_DIR}/data_aligned"

def analyze_dataset():
    print("=" * 60)
    print("📊 数据集分布统计")
    print("=" * 60)
    
    subsets = ["UTKFace", "AFAD", "imdb"]
    all_files = []
    all_ages = []
    all_genders = []
    
    for subset in subsets:
        subset_path = os.path.join(DATA_DIR, subset)
        if not os.path.exists(subset_path):
            print(f"⚠️ 警告：{subset} 路径不存在")
            continue
        
        files = [f for f in os.listdir(subset_path) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\n📍 {subset}:")
        print(f"   样本数: {len(files)}")
        
        for f in files:
            try:
                parts = f.split('_')
                age = float(parts[0])
                gender = int(parts[1])
                
                all_files.append(os.path.join(subset, f))
                all_ages.append(age)
                all_genders.append(gender)
            except:
                continue
    
    total = len(all_files)
    if total == 0:
        print("\n❌ 错误：未找到任何数据！")
        return
    
    print("\n" + "=" * 60)
    print("📈 总体统计")
    print("=" * 60)
    
    male_count = sum(1 for g in all_genders if g == 0)
    female_count = sum(1 for g in all_genders if g == 1)
    
    print(f"\n总样本数: {total}")
    print(f"训练集: {int(total * 0.8)}")
    print(f"测试集: {int(total * 0.2)}")
    print(f"\n性别比例:")
    print(f"  男性: {male_count} ({male_count/total*100:.1f}%)")
    print(f"  女性: {female_count} ({female_count/total*100:.1f}%)")
    
    print(f"\n年龄跨度分布:")
    age_ranges = [(0, 20), (21, 40), (41, 60), (61, 200)]
    age_labels = ["0-20岁", "21-40岁", "41-60岁", "60+岁"]
    
    for (min_age, max_age), label in zip(age_ranges, age_labels):
        count = sum(1 for a in all_ages if min_age <= a <= max_age)
        print(f"  {label}: {count}")
    
    print(f"\n平均年龄: {np.mean(all_ages):.1f} 岁")
    print(f"年龄范围: {min(all_ages):.0f} - {max(all_ages):.0f} 岁")
    
    print("\n" + "=" * 60)
    print("✅ 统计完成！请将上述数据填入研究报告第五章。")
    print("=" * 60)

if __name__ == "__main__":
    analyze_dataset()
