import torch
from torch.utils import data
import numpy as np
import math
import os
import SimpleITK as sitk
from tqdm import tqdm
import lmdb
import pickle
from PIL import Image
from dataloader.ms_prostate.normalize import normalize_image_to_m1_1
from skimage.transform import resize


def _is_nifti(path):
    return path.endswith('.nii') or path.endswith('.nii.gz')

def _abspath(root, p):
    return p if os.path.isabs(p) else os.path.join(root, p)

def _basename_wo_nii(p):
    base = os.path.basename(p)
    if base.endswith('.nii.gz'):
        return base[:-7]
    if base.endswith('.nii'):
        return base[:-4]
    return os.path.splitext(base)[0]


class PROSTATE_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=384, batch_size=None,
                 img_normalize=True, transform=None, aug_wt=1.0):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.aug_wt = aug_wt
        self.image_pool, self.label_pool, self.name_pool, self.key_pool = [], [], [], []
        self.transform = transform

        self._read_img_into_memory()

        if batch_size is not None:
            iter_nums = len(self.image_pool) // batch_size
            if iter_nums == 0:
                iter_nums = 1
            scale = math.ceil(250 / iter_nums)
            self.image_pool *= scale
            self.label_pool *= scale
            self.name_pool  *= scale
            self.key_pool   *= scale

        print('Image Nums (volumes):', len(self.img_list))
        print('Slice Nums (samples):', len(self.image_pool))

        self.lmdb = None
        self.lmdb_translation = None

        if len(self.image_pool) == 0:
            raise RuntimeError(
                "[Dataset Empty inside PROSTATE_dataset] 没有任何切片被加入。请检查：\n"
                "- CSV 是否正确\n"
                "- NIfTI 路径是否存在\n"
                "- 掩码是否正确\n"
                "- data.lmdb 是否存在对应 key"
            )

    def __len__(self):
        return len(self.image_pool)

    def __getitem__(self, item):
        if self.lmdb is None:
            self.lmdb = lmdb.open(
                os.path.join(self.root, "data.lmdb"),
                readonly=True, lock=False, readahead=False
            )
        if self.lmdb_translation is None:
            self.lmdb_translation = lmdb.open(
                os.path.join(self.root, "data.lmdb"),
                readonly=True, lock=False, readahead=False
            )

        fname = self.name_pool[item]
        key   = self.key_pool[item]

        def to_2d(x):
            if x.ndim == 2:
                return x
            if x.ndim == 3:
                if x.shape[0] in (1, 3):
                    return x[0]
                if x.shape[2] == 1:
                    return x[..., 0]
            x2 = np.squeeze(x)
            assert x2.ndim == 2, f"Expect 2D for albumentations, got {x.shape}"
            return x2

        if _is_nifti(fname):
            img_npy, label_npy, name = self.load_data_point(self.lmdb, key)
            if self.img_normalize:
                img_npy = normalize_image_to_m1_1(img_npy)
            label_npy = (label_npy > 0).astype(np.uint8)

            img2d = to_2d(img_npy).astype(np.float32)
            lbl2d = to_2d(label_npy).astype(np.uint8)

            if self.transform is not None:
                t = self.transform(image=img2d, mask=lbl2d)
                img2d, lbl2d = t["image"], t["mask"]

            img_chw = np.stack([img2d]*3, axis=0)
            lbl_chw = lbl2d[None, ...].astype(np.uint8)

            sample = {
                "image": torch.from_numpy(img_chw).float(),
                "label": torch.from_numpy(lbl_chw).float(),
                "img_name": name,
                "aug_wt": self.aug_wt,
            }
            return sample

        elif fname.endswith(".png"):
            img_npy, label_npy, name = self.load_data_point(self.lmdb_translation, key)

            # 强制灰度化 image
            if img_npy.ndim == 3 and img_npy.shape[2] == 3:
                img_gray = np.mean(img_npy, axis=2).astype(np.float32)
            elif img_npy.ndim == 2:
                img_gray = img_npy.astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected PNG shape: {img_npy.shape}")

            # ⚠️ 保证 label 单通道
            if label_npy.ndim == 3 and label_npy.shape[2] == 3:
                label_npy = label_npy[..., 0]  # 只取一个通道
            elif label_npy.ndim == 2:
                label_npy = label_npy.astype(np.uint8)
            else:
                raise RuntimeError(f"Unexpected label PNG shape: {label_npy.shape}")

            # resize
            if img_gray.shape != self.target_size:
                img_gray = resize(img_gray, self.target_size, order=1, preserve_range=True).astype(np.float32)
            if label_npy.shape != self.target_size:
                label_npy = resize(label_npy, self.target_size, order=0, preserve_range=True).astype(np.uint8)

            if self.img_normalize:
                img_gray = normalize_image_to_m1_1(img_gray)
            label_npy = (label_npy > 0).astype(np.uint8)

            if self.transform is not None:
                t = self.transform(image=img_gray, mask=label_npy)
                img_gray, label_npy = t["image"], t["mask"]

            # 保持和 NIfTI 一致
            img_chw = np.stack([img_gray]*3, axis=0)          # (3,H,W)
            lbl_chw = label_npy[None, ...].astype(np.uint8)   # (1,H,W)

            sample = {
                "image": torch.from_numpy(img_chw).float(),
                "label": torch.from_numpy(lbl_chw).float(),
                "img_name": name,
                "aug_wt": self.aug_wt,
            }
            return sample


        else:
            raise RuntimeError(f"Unsupported file type for item {fname}")

    def _read_img_into_memory(self):
        img_num = len(self.img_list)
        for index in tqdm(range(img_num)):
            p_img = _abspath(self.root, self.img_list[index])
            p_msk = _abspath(self.root, self.label_list[index])

            if _is_nifti(p_img):
                img_sitk = sitk.ReadImage(p_img)
                msk_sitk = sitk.ReadImage(p_msk)
                img_npy = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
                msk_npy = sitk.GetArrayFromImage(msk_sitk).astype(np.uint8)

                D = img_npy.shape[0]

                # ✅ 去掉 split，不再用 train/val/test
                domain = os.path.basename(os.path.dirname(os.path.dirname(p_img)))   # BIDMC
                case_id = _basename_wo_nii(p_img)
                key_prefix = f"{domain}_{case_id}"

                for sl in range(D):
                    if msk_npy[sl].max() == 0:   # ⚠ 没有前景 → 跳过
                        continue
                    self.image_pool.append((p_img, sl))
                    self.label_pool.append((p_msk, sl))
                    self.name_pool.append(p_img)
                    self.key_pool.append(f"{key_prefix}_{sl}")

            elif p_img.endswith('.png'):
                img_path = p_img
                dom_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                dom_id = dom_id.replace("Domain", "")
                fname = os.path.splitext(os.path.basename(img_path))[0]
                key = f"{dom_id}_{fname}"

                self.image_pool.append(img_path)
                self.label_pool.append(p_msk)
                self.name_pool.append(img_path)
                self.key_pool.append(key)

            else:
                raise RuntimeError(f"Unsupported file type: {p_img}")

    def deserialize_data(self, serialized_data):
        data = pickle.loads(serialized_data)
        return data['image'], data['label'], data['name']

    def load_data_point(self, env, key):
        with env.begin() as txn:
            serialized_data = txn.get(key.encode('utf-8'))
            if serialized_data:
                return self.deserialize_data(serialized_data)
            else:
                raise KeyError(f'{key} not found')
