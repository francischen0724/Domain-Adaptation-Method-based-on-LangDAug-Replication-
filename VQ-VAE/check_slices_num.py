# check_batch_shapes.py
import torch
from torch.utils.data import DataLoader
from dataloader.ms_prostate.PROSTATE_dataloader import PROSTATE_dataset
from dataloader.ms_prostate.convert_csv_to_list import convert_labeled_list
from dataloader.ms_prostate.transform import collate_fn_w_transform

if __name__ == "__main__":
    # 数据集根目录
    dataset_root = "/project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/LangDAug/VQ-VAE/datasets/prostate"

    # 使用哪些 CSV 文件
    source_name = ["BIDMC"]   # 你也可以换成 ["BIDMC","BMC",...] 
    source_csv = [s + "_train.csv" for s in source_name]

    # 读取 CSV -> img_list, label_list
    sr_img_list, sr_label_list = convert_labeled_list(dataset_root, source_csv)

    # 创建 dataset 和 dataloader
    train_dataset = PROSTATE_dataset(
        dataset_root,
        sr_img_list,
        sr_label_list,
        target_size=384,
        batch_size=8,             # 随便一个 batch size
        img_normalize=True,
        transform=None            # 不做数据增强，方便检查 shape
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn_w_transform,
        num_workers=0             # 建议先设0，避免多进程调试麻烦
    )

    # 检查前 3 个 batch
    for i, batch in enumerate(train_dataloader):
        imgs, masks = batch["data"], batch["mask"]
        print(f"[Batch {i}] image={imgs.shape}, label={masks.shape}, "
              f"image min/max=({imgs.min().item():.3f}, {imgs.max().item():.3f}), "
              f"label unique={torch.unique(masks)}")
        if i == 2:
            break
