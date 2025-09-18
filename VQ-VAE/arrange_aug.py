import os
import shutil

# ===== 配置 =====
src_root = "datasets/fundus/train_LAB_LD_234/Domain423"       # 原数据根目录
dst_root = "datasets/fundus/train_LAB_LD_all13/Domain423"    # 筛选结果根目录

# 创建目标目录结构
for sub in ["image", "mask"]:
    os.makedirs(os.path.join(dst_root, sub), exist_ok=True)

count = 0
# 遍历 image 文件夹
src_img_dir = os.path.join(src_root, "image")
src_mask_dir = os.path.join(src_root, "mask")
dst_img_dir = os.path.join(dst_root, "image")
dst_mask_dir = os.path.join(dst_root, "mask")

for fname in os.listdir(src_img_dir):
    img_src_path = os.path.join(src_img_dir, fname)
    mask_src_path = os.path.join(src_mask_dir, fname)
    # 确保 mask 存在
    if os.path.exists(mask_src_path):
        shutil.copy(img_src_path, dst_img_dir)
        shutil.copy(mask_src_path, dst_mask_dir)
        count += 1
    else:
        print(f"⚠️ 缺少 mask：{fname}")

print(f"✅ 共复制 {count} 对 image/mask 到 {dst_root}")
