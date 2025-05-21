import torch
from torch.utils import data
import numpy as np
import math
import os
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from dataloader.ms_prostate.normalize import normalize_image, normalize_image_to_m1_1
from dataloader.ms_prostate.convert_csv_to_list import convert_labeled_list
from dataloader.ms_prostate.transform import collate_fn_w_transform
from torch.utils.data import DataLoader
from tqdm import tqdm
import lmdb
import pickle

class PROSTATE_dataset(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=384, batch_size=None, img_normalize=True, transform=None):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.target_size = (target_size, target_size)
        self.img_normalize = img_normalize
        self.image_pool, self.label_pool, self.name_pool, self.key_pool = [], [], [], []
        self._read_img_into_memory()
        if batch_size is not None:
            iter_nums = len(self.image_pool) // batch_size
            scale = math.ceil(250 / iter_nums)
            self.image_pool = self.image_pool * scale
            self.label_pool = self.label_pool * scale
            self.name_pool = self.name_pool * scale
            self.key_pool = self.key_pool * scale
        self.transform = transform

        print('Image Nums:', len(self.img_list))
        print('Slice Nums:', len(self.image_pool))

        self.lmdb = lmdb.open(os.path.join(self.root, 'data.lmdb'), readonly=True, lock=False)


    def __len__(self):
        return len(self.image_pool)

    def __getitem__(self, item):

        img_npy, label_npy, name = self.load_data_point(self.key_pool[item])

        if self.img_normalize:
            img_npy = normalize_image_to_m1_1(img_npy)
        label_npy[label_npy > 1] = 1

        sample = {'image': img_npy, 'label': label_npy, 'img_name': name}

        if self.transform is not None:
            transformed = self.transform(image=sample['image'][0], mask=sample['label'][0])

            sample['image'][0], sample['label'][0] = transformed['image'], transformed['mask']

        sample['image'] = np.repeat(sample['image'], 3, axis=0)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['label'] = torch.from_numpy(sample['label']).float()

        return sample

    def _read_img_into_memory(self):
        img_num = len(self.img_list)
        for index in tqdm(range(img_num)):
            img_file = os.path.join(self.root, self.img_list[index])
            label_file = os.path.join(self.root, self.label_list[index])

            img_sitk = sitk.ReadImage(img_file)
            label_sitk = sitk.ReadImage(label_file)

            img_npy = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
            label_npy = sitk.GetArrayFromImage(label_sitk)

            for slice in range(img_npy.shape[0]):
                if label_npy[slice, :, :].max() > 0:
                    dir_name = os.path.basename(os.path.dirname(img_file))
                    img_basename = os.path.basename(img_file).split('.')[0]
                    self.image_pool.append((img_file, slice))
                    self.label_pool.append((label_file, slice))
                    self.name_pool.append(img_file)
                    self.key_pool.append(f'{dir_name}_{img_basename}_{slice}')

    def preprocess(self, x):
        mask = x > 0
        y = x[mask]

        lower = np.percentile(y, 0.2)
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper
        return np.expand_dims(x, axis=0)
    
    def deserialize_data(self, serialized_data):
        data = pickle.loads(serialized_data)
        return data['image'], data['label'], data['name']

    def load_data_point(self, key):
        with self.lmdb.begin() as txn:
            serialized_data = txn.get(key.encode('utf-8'))
            if serialized_data:
                return self.deserialize_data(serialized_data)
            else:
                raise KeyError(f'{key} not found')

if __name__ == '__main__':

    dataset_root = 'provide/your/path'
    image_size = 384
    batch_size = 8
    source_name = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
    source_csv = []
    for s_n in source_name:
        source_csv.append(s_n + '.csv')
    sr_img_list, sr_label_list = convert_labeled_list(dataset_root, source_csv)
    train_dataset = PROSTATE_dataset(dataset_root, sr_img_list, sr_label_list,
                                            image_size, batch_size, img_normalize=False)
    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    collate_fn=collate_fn_w_transform,
                                    num_workers=4)
    
    for i, batch in enumerate(train_dataloader):
        print(i)
        image, mask = batch['data'], batch['mask']
        print(image.shape, mask.shape, torch.max(image), torch.min(image), torch.max(mask), torch.min(mask), torch.unique(mask))