from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader
from dataloaders.dataloader.ms_fundus.fundus_dataloader import FundusSegmentation
from dataloaders.dataloader.ms_fundus import fundus_transforms as tr
from dataloaders.dataloader.ms_prostate.PROSTATE_dataloader import PROSTATE_dataset
from dataloaders.dataloader.ms_prostate.convert_csv_to_list import convert_labeled_list
from torchvision import transforms, utils
import torch
import os
from glob import glob
import albumentations as A
import cv2

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'coco':
        train_set = coco.COCOSegmentation(args, split='train')
        val_set = coco.COCOSegmentation(args, split='val')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'fundus':
        data_root = args.data_root
        if not args.testing:
            train_set = FundusSegmentation(args, base_dir=data_root, phase='train', splitid=args.splitid)
            val_set = FundusSegmentation(args, base_dir=data_root, phase='test', splitid=args.valid)

        test_set = FundusSegmentation(args, base_dir=data_root, phase='test', splitid=args.testid)
        
        if not args.testing:
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True, **kwargs)
        else:
            num_class = 2
            train_loader = None
            val_loader = None 
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    
    elif args.dataset == 'prostate':
        data_root = args.data_root
        source_csv, target_csv = [], []
        sr_tr_img_list = []
        sr_tr_label_list = []

        composed_transforms_tr = A.Compose([											
											A.RandomSizedCrop(min_max_height=(300,330), height=384, width=384, p=0.3),
										])
        for id in args.splitid:
            if id in ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']:
                source_csv.append(id + '.csv')
            else:
                image_dir = os.path.join(data_root, 'train_LAB_LD_prostate', 'Domain'+id, 'image/')
                image_list = glob(image_dir + '*.png')
                sr_tr_img_list += image_list

        for id in args.testid:
            target_csv.append(id + '.csv')

        for image_path in sr_tr_img_list:
            label_path = image_path.replace('image', 'mask')
            sr_tr_label_list.append(label_path)

        sr_img_list, sr_label_list = convert_labeled_list(data_root, source_csv)
        tr_img_list, tr_label_list = convert_labeled_list(data_root, target_csv)

        sr_img_list += sr_tr_img_list
        sr_label_list += sr_tr_label_list

        train_set = PROSTATE_dataset(data_root, sr_img_list, sr_label_list, 384, args.batch_size, img_normalize=True, transform=composed_transforms_tr, aug_wt=args.aug_wt)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12, drop_last=True)
        val_set = PROSTATE_dataset(data_root, tr_img_list, tr_label_list, 384, 1, img_normalize=True, transform=composed_transforms_tr)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)
        test_set = PROSTATE_dataset(data_root, tr_img_list, tr_label_list, 384, 1, img_normalize=True, transform=None)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)
        num_class = 1

        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

