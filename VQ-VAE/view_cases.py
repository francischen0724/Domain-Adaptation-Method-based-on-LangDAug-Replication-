import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataloader.ms_prostate.PROSTATE_dataloader import PROSTATE_dataset
from dataloader.ms_prostate.convert_csv_to_list import convert_labeled_list

# ===== 配置 =====
data_root = "datasets/prostate"
csv_list = ["UCL_train.csv"]  # CSV 文件
output_dir = "outputs_slices/UCL"
max_cases = 2   # 只输出前 2 个病例

# ===== 创建 dataset =====
img_list, label_list = convert_labeled_list(data_root, csv_list)

dataset = PROSTATE_dataset(
    root=data_root,
    img_list=img_list,
    label_list=label_list,
    target_size=384,
    batch_size=1,
    img_normalize=True,
    transform=None
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ===== 输出目录 =====
os.makedirs(output_dir, exist_ok=True)

# ===== 保存样本 =====
case_seen = {}

for idx, sample in enumerate(loader):
    # e.g. "image_Case07_15.png" -> "Case07"
    case_name = sample["img_name"][0].split("_")[1]

    # 如果出现新病例，且数量已达上限，就退出循环
    if case_name not in case_seen and len(case_seen) >= max_cases:
        break

    if case_name not in case_seen:
        case_seen[case_name] = 0
    case_seen[case_name] += 1

    img = sample["image"]  # (1,3,H,W), [-1,1]
    lbl = sample["label"]  # (1,1,H,W)
    lbl_rgb = lbl.repeat(1, 3, 1, 1).float()

    # 拼接 原图 + 叠加mask
    combined = torch.cat([img, img * 0.7 + lbl_rgb * 0.3], dim=0)

    # 保存到对应病例文件夹
    case_out_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_out_dir, exist_ok=True)
    out_path = os.path.join(case_out_dir, f"{sample['img_name'][0]}.png")

    save_image(
        combined,
        out_path,
        nrow=2,
        normalize=True,
        value_range=(-1, 1)
    )
    print(f"✅ 已保存 {out_path}")

print("\n保存完成：")
for case, num in case_seen.items():
    print(f"{case}: {num} slices")
