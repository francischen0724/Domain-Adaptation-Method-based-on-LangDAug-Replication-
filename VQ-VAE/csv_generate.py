# generate_prostate_csv.py
import os
import pandas as pd
from pathlib import Path

dataroot = "./datasets/prostate"
splits = ["train", "val", "test"]
domains = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]

for split in splits:
    for domain in domains:
        img_dir = Path(dataroot) / split / domain / "image"
        msk_dir = Path(dataroot) / split / domain / "mask"

        if not img_dir.exists() or not msk_dir.exists():
            print(f"⚠️ Skip {domain} {split}, no folder found")
            continue

        img_list = sorted([f for f in os.listdir(img_dir) if f.endswith((".nii", ".nii.gz", ".png"))])
        msk_list = sorted([f for f in os.listdir(msk_dir) if f.endswith((".nii", ".nii.gz", ".png"))])

        if len(img_list) != len(msk_list):
            print(f"⚠️ Mismatch in {domain} {split}: {len(img_list)} images vs {len(msk_list)} masks")

        pairs = []
        for i in range(len(img_list)):
            img_path = os.path.join(split, domain, "image", img_list[i])
            msk_path = os.path.join(split, domain, "mask", msk_list[i])
            pairs.append({"image": img_path, "mask": msk_path})

        if pairs:
            out_csv = Path(dataroot) / f"{domain}_{split}.csv"
            pd.DataFrame(pairs).to_csv(out_csv, index=False)
            print(f"✅ Saved {out_csv}, total {len(pairs)} pairs")
