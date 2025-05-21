import nibabel as nib
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
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
from dataloader.ms_prostate.PROSTATE_dataloader import PROSTATE_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import lmdb
import pickle
from collections import OrderedDict
from glob import glob

dataset_root = '/provide/your/path'
translated_data_root = os.path.join(dataset_root, 'train_LAB_LD_prostate')
pair_paths = glob(translated_data_root + '/*')
pairs = []
for pair in pair_paths:
    pair = pair.split('/')[-1].split('Domain')[-1]
    pairs.append(pair)

def serialize_data(image, label, name):
    return pickle.dumps({'image': image, 'label': label, 'name': name})

def save_data_point(env, key, image, label, name):
    with env.begin(write=True) as txn:
        serialized_data = serialize_data(image, label, name)
        txn.put(key.encode('utf-8'), serialized_data)

def deserialize_data(serialized_data):
    data = pickle.loads(serialized_data)
    return data['image'], data['label'], data['name']

def load_data_point(env, key):
    with env.begin() as txn:
        serialized_data = txn.get(key.encode('utf-8'))
        if serialized_data:
            return deserialize_data(serialized_data)
        else:
            raise KeyError(f'{key} not found')
        
class LimitedQueueCache:
    def __init__(self, max_size=10, purge_count=5):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.purge_count = purge_count

    def __setitem__(self, key, value):
        if key in self.cache:
            # Move the existing key to the end to mark it as most recently used
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self._purge()

    def __getitem__(self, key):
        if key in self.cache:
            # Move the accessed key to the end to mark it as most recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return None

    def __delitem__(self, key):
        if key in self.cache:
            del self.cache[key]
        else:
            raise KeyError(f'Key {key} not found in cache')

    def _purge(self):
        # Remove the oldest `purge_count` items
        for _ in range(self.purge_count):
            if self.cache:
                self.cache.popitem(last=False)

    def __contains__(self, key):
        return key in self.cache

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self.cache.items())})'


lmdb_path = os.path.join('/provide/path', 'data_translated.lmdb')


img_path_list = []
label_path_list = []
key_list = []

for i, pair_path in enumerate(tqdm(pair_paths)):
    img_dir = os.path.join(pair_path, 'image')
    img_paths = glob(img_dir + '/*')
    for img_path in tqdm(img_paths):
        img_path_list.append(img_path)
        label_path_list.append(img_path.replace('image', 'mask'))
        key_list.append(f'{pairs[i]}_{img_path.split("/")[-1].split(".")[0]}')

print(len(img_path_list), len(label_path_list), len(key_list))

for i in range(10):
    print(img_path_list[i], label_path_list[i], key_list[i])

env = lmdb.open(lmdb_path, map_size=int(350e9))

for i, img_path in enumerate(tqdm(img_path_list)):
    img_npy = np.asarray(Image.open(img_path))
    label_npy = np.asarray(Image.open(label_path_list[i]))

    save_data_point(env, key_list[i], img_npy, label_npy, key_list[i])

env.close()