#!/usr/bin/env python3
# count_fg_slices.py
import os, csv, argparse
import numpy as np
import SimpleITK as sitk
from glob import glob
from collections import defaultdict

def basename_nii(path):
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]

def domain_from_relpath(relpath):
    # relpath 形如: train/BIDMC/mask/CaseXX_segmentation.nii
    # 或 train/BIDMC/image/CaseXX.nii
    parts = relpath.replace("\\", "/").split("/")
    # 期待 [..., split, DOMAIN, ...]
    if len(parts) >= 2:
        return parts[1]
    return "UNKNOWN"

def count_case(mask_path):
    m = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    if m.ndim == 3:
        Z = m.shape[0]
        fg_per_slice = (m.reshape(Z, -1) > 0).any(axis=1)
        total = int(Z)
        fg = int(fg_per_slice.sum())
    elif m.ndim == 2:
        total = 1
        fg = int((m > 0).any())
    else:
        raise RuntimeError(f"Unexpected mask ndim={m.ndim} for {mask_path}")
    return total, fg

def scan_split(root, split, show_cases=False):
    root = os.path.abspath(root)
    csv_paths = sorted(glob(os.path.join(root, f"*_{split}.csv")))
    if not csv_paths:
        print(f"[{split}] 未找到 *_${split}.csv")
        return

    dom_stats = defaultdict(lambda: {"vols": 0, "total_slices": 0, "fg_slices": 0})
    case_rows = []

    for csv_path in csv_paths:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel = row["image"]
                msk_rel = row["mask"]
                img_path = os.path.join(root, img_rel)
                msk_path = os.path.join(root, msk_rel)

                if not os.path.exists(msk_path):
                    print(f"⚠ 缺失mask: {msk_path}, 跳过")
                    continue
                dom = domain_from_relpath(msk_rel)
                case = basename_nii(img_path)

                total, fg = count_case(msk_path)

                dom_stats[dom]["vols"] += 1
                dom_stats[dom]["total_slices"] += total
                dom_stats[dom]["fg_slices"] += fg

                if show_cases:
                    case_rows.append((split, dom, case, total, fg))

    # 打印汇总
    print(f"\n=== [{split}] 按 domain 的统计 ===")
    s_vols = s_total = s_fg = 0
    for dom in sorted(dom_stats.keys()):
        d = dom_stats[dom]
        s_vols += d["vols"]; s_total += d["total_slices"]; s_fg += d["fg_slices"]
        ratio = (d["fg_slices"] / d["total_slices"] * 100.0) if d["total_slices"] else 0.0
        print(f"- {dom:6s} | vols={d['vols']:4d} | total_slices={d['total_slices']:5d} | fg_slices={d['fg_slices']:5d} | fg_ratio={ratio:6.2f}%")
    if dom_stats:
        ratio_all = (s_fg / s_total * 100.0) if s_total else 0.0
        print(f"TOTAL    | vols={s_vols:4d} | total_slices={s_total:5d} | fg_slices={s_fg:5d} | fg_ratio={ratio_all:6.2f}%")

    # 打印每个 case（可选）
    if show_cases and case_rows:
        print(f"\n--- [{split}] 每个 case 统计 ---")
        for row in sorted(case_rows):
            sp, dom, case, total, fg = row
            ratio = (fg / total * 100.0) if total else 0.0
            print(f"{dom:6s} | {case:16s} | total={total:4d} | fg={fg:4d} | fg_ratio={ratio:6.2f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="prostate 数据集根目录（包含 train/val/test 和 CSV）")
    ap.add_argument("--splits", default="train,val,test", help="要统计的split，逗号分隔")
    ap.add_argument("--show-cases", action="store_true", help="打印每个 case 的明细")
    args = ap.parse_args()

    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        scan_split(args.root, sp, show_cases=args.show_cases)

if __name__ == "__main__":
    main()
