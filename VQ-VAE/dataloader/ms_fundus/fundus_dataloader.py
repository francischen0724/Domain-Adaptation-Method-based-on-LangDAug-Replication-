from __future__ import print_function, division
import os
import sys
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import copy
import torch


class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 4 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='./datasets/fundus/test/',
                 phase='train',
                 splitid=[1, 2, 3],
                 transform=None,
                 state='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.image_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.label_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}
        self.img_name_pool = {'DGS':[], 'REF':[], 'RIM':[], 'REF_val':[]}

        self.img_extension = {1: '.png', 2: '.jpg', 3: '.jpg', 4: '.jpg'}
        self.label_extension = {1: '.png', 2: '.png', 3: '.bmp', 4: '.bmp'}

        self.flags_DGS = ['gd', 'nd']
        self.flags_REF = ['g', 'n']
        self.flags_RIM = ['G', 'N', 'S']
        self.flags_REF_val = ['V']
        self.splitid = splitid
        SEED = 1212
        random.seed(SEED)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir,  phase, 'Domain'+str(id), 'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*' + self.img_extension[id])
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                gt_path = gt_path.replace(self.img_extension[id], self.label_extension[id])
                self.image_list.append({'image': image_path, 'label': gt_path})

        print(len(self.image_list))
        random.shuffle(self.image_list)
        print(len(self.image_list))

        self.image_pool_raw = []
        self.label_pool_raw = []
        self.img_name_pool_raw = []
        self.domain_code_raw = []

        self.transform = transform
        self._read_img_into_memory()
        
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break

        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):


        _img = self.image_pool_raw[index]
        _target = self.label_pool_raw[index]
        _img_name = self.img_name_pool_raw[index]
        _dc = self.domain_code_raw[index]

        sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': _dc}

        if self.transform is not None:
            sample = self.transform(sample)
        sample['label'] = torch.cat((sample['label'], sample['label'], sample['label']), 0)

        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_DGS:
                Flag = 'DGS'
            elif basename[0] in self.flags_REF:
                Flag = 'REF'
            elif basename[0] in self.flags_RIM:
                Flag = 'RIM'
            elif basename[0] in self.flags_REF_val:
                Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            
            self.domain_code_raw.append( list(self.image_pool.keys()).index(Flag) )
            
            if Flag=='RIM': 
                self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').crop((5, 0, 2144/2, 1424)).resize((256, 256), Image.LANCZOS))
                self.image_pool_raw.append(Image.open(self.image_list[index]['image']).convert('RGB').crop((5, 0, 2144/2, 1424)).resize((256, 256), Image.LANCZOS))

                _target = np.asarray(Image.open(self.image_list[index]['label']).convert('L'))
                _target = _target[144:144+512, 144:144+512]
                _target = Image.fromarray(_target)
            else:
                self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                self.image_pool_raw.append(Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))

                _target = Image.open(self.image_list[index]['label'])

            if _target.mode is 'RGB':
                _target = _target.convert('L')
            if self.state != 'prediction':
                _target = _target.resize((256, 256))

            self.label_pool[Flag].append(_target)
            self.label_pool_raw.append(_target)

            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)
            self.img_name_pool_raw.append(_img_name)

    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(10 - sum(self.splitid)) + ')'


if __name__ == '__main__':
    import dataloader.ms_fundus.fundus_transforms as tr
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomSizedCrop(512),
        tr.RandomRotate(15),
        tr.Normalize_tf(),
        tr.ToTensor()])

    voc_train = FundusSegmentation(phase='train', splitid=[1],
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample[0]["image"].size()[0]):
            img = sample[0]['image'].numpy()
            gt = sample[0]['label'].numpy()
            segmap = np.transpose(gt[jj], axes=[1, 2, 0]).astype(np.uint8)
            img_tmp = np.transpose((img[jj]+1.0)*128, axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(221)
            plt.imshow(img_tmp)
            plt.subplot(222)
            plt.imshow(segmap[..., 0])
            plt.subplot(223)
            plt.imshow(segmap[..., 1])

            break
    plt.show(block=True)
