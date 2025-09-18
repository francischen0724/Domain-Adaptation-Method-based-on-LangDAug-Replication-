import os, lmdb, pickle
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import argparse
from PIL import Image

def dumps(img, lbl, name):
    return pickle.dumps({"image": img.astype(np.float32),
                         "label": lbl.astype(np.uint8),
                         "name": name}, protocol=pickle.HIGHEST_PROTOCOL)

def basename_nii(path):
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]

def write_from_split(env, root, split, commit_every=4096):
    """写入 train/val/test 文件夹内的所有 domain (NIfTI)"""
    total = 0
    split_dir = os.path.join(root, split)
    if not os.path.exists(split_dir):
        print(f"⚠ 没找到 {split_dir}，跳过 {split}")
        return 0

    with env.begin(write=True) as txn:
        for domain in sorted(os.listdir(split_dir)):
            dom_path = os.path.join(split_dir, domain)
            if not os.path.isdir(dom_path): 
                continue

            img_dir = os.path.join(dom_path, "image")
            mask_dir = os.path.join(dom_path, "mask")
            nii_imgs = sorted(glob(os.path.join(img_dir, "*.nii*")))
            nii_msks = sorted(glob(os.path.join(mask_dir, "*.nii*")))
            if not nii_imgs:
                continue

            print(f"➡ 写入 domain={domain}, 共 {len(nii_imgs)} 个 case")

            for ip, mp in zip(nii_imgs, nii_msks):
                img = sitk.GetArrayFromImage(sitk.ReadImage(ip)).astype(np.float32)  # (Z,H,W)
                msk = sitk.GetArrayFromImage(sitk.ReadImage(mp)).astype(np.uint8)

                case_id = basename_nii(ip)   # e.g. Case05
                key_prefix = f"{domain}_{case_id}"  # ✅ 唯一标识

                Z = img.shape[0]
                for z in range(Z):
                     # ✅ 只保留有前景的切片
                    if msk[z].max() > 0:
                        key = f"{key_prefix}_{z}"  # train_BIDMC_Case05_23
                        txn.put(key.encode("utf-8"), dumps(img[z], msk[z], key))
                        total += 1
                        if (total % commit_every) == 0:
                            txn.commit(); txn = env.begin(write=True)

    return total


def write_from_png(env, root, commit_every=4096):
    """LAB_LD PNG 数据"""
    total = 0
    ld_root = os.path.join(root, "train_LAB_LD_prostate")
    if not os.path.exists(ld_root):
        print(f"⚠ 没找到 {ld_root}，跳过 LAB_LD 写入")
        return 0

    txn = env.begin(write=True)
    for domain in sorted(glob(os.path.join(ld_root, "Domain*"))):
        dom_id = os.path.basename(domain).replace("Domain", "")  # e.g. "221"
        img_dir = os.path.join(domain, "image")
        for ip in tqdm(sorted(glob(os.path.join(img_dir, "*.png"))), desc=f"➡ 写入 {dom_id}"):
            mp = ip.replace(os.sep+"image"+os.sep, os.sep+"mask"+os.sep)
            if not os.path.exists(mp):
                print(f"⚠ {mp} 缺失"); continue

            img = np.array(Image.open(ip).convert("L"), dtype=np.float32)  # H,W
            msk = np.array(Image.open(mp), dtype=np.uint8)                 # H,W

            name = os.path.splitext(os.path.basename(ip))[0]  # e.g. Case01_39_4
            key = f"{dom_id}_{name}"  # ✅ LAB_LD 用 domain id

            txn.put(key.encode("utf-8"), dumps(img, msk, key))
            total += 1
            if (total % commit_every) == 0:
                txn.commit(); txn = env.begin(write=True)

    txn.commit()
    return total


def main(root, splits=("train","val","test"), map_size_gb=64, commit_every=4096):
    root = os.path.abspath(root)
    lmdb_path = os.path.join(root, "data.lmdb")
    os.makedirs(root, exist_ok=True)

    env = lmdb.open(lmdb_path, map_size=int(map_size_gb * (1024**3)))

    # 1. 原始 train/val/test 结构
    total_csv = 0
    for sp in splits:
        total_csv += write_from_split(env, root, sp, commit_every=commit_every)

    # 2. LAB_LD PNG
    total_png = write_from_png(env, root, commit_every=commit_every)

    env.sync(); env.close()
    print(f"✅ 写入完成：{lmdb_path}，原始切片 {total_csv} 张，LAB_LD 切片 {total_png} 张，总计 {total_csv+total_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="prostate 数据集根目录，例如 /.../datasets/prostate")
    ap.add_argument("--splits", type=str, default="train,val,test",
                    help="逗号分隔，可选：train,val,test")
    ap.add_argument("--map_size_gb", type=int, default=64,
                    help="LMDB map_size，默认64GB，按需调大")
    ap.add_argument("--commit_every", type=int, default=4096,
                    help="每多少切片提交一次事务")
    args = ap.parse_args()
    main(args.root, tuple(s.strip() for s in args.splits.split(",")),
         map_size_gb=args.map_size_gb, commit_every=args.commit_every)
