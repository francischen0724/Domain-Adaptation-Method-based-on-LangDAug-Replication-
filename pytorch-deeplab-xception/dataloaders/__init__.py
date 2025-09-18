# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
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
            val_set = FundusSegmentation(args, base_dir=data_root, phase='val', splitid=args.valid)

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
    
    # elif args.dataset == 'prostate':
    #     data_root = args.data_root
    #     source_csv, target_csv = [], []
    #     sr_tr_img_list = []
    #     sr_tr_label_list = []

    #     composed_transforms_tr = A.Compose([											
	# 										A.RandomSizedCrop(min_max_height=(300,330), height=384, width=384, p=0.3),
	# 									])
    #     for id in args.splitid:
    #         if id in ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']:
    #             source_csv.append(id + '.csv')
    #         else:
    #             image_dir = os.path.join(data_root, 'train_LAB_LD_prostate', 'Domain'+id, 'image/')
    #             image_list = glob(image_dir + '*.png')
    #             sr_tr_img_list += image_list

    #     for id in args.testid:
    #         target_csv.append(id + '.csv')

    #     for image_path in sr_tr_img_list:
    #         label_path = image_path.replace('image', 'mask')
    #         sr_tr_label_list.append(label_path)

    #     sr_img_list, sr_label_list = convert_labeled_list(data_root, source_csv)
    #     tr_img_list, tr_label_list = convert_labeled_list(data_root, target_csv)

    #     sr_img_list += sr_tr_img_list
    #     sr_label_list += sr_tr_label_list

    #     train_set = PROSTATE_dataset(data_root, sr_img_list, sr_label_list, 384, args.batch_size, img_normalize=True, transform=composed_transforms_tr, aug_wt=args.aug_wt)
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12, drop_last=True)
    #     val_set = PROSTATE_dataset(data_root, tr_img_list, tr_label_list, 384, 1, img_normalize=True, transform=composed_transforms_tr)
    #     val_loader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)
    #     test_set = PROSTATE_dataset(data_root, tr_img_list, tr_label_list, 384, 1, img_normalize=True, transform=None)
    #     test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=0, drop_last=True)
    #     num_class = 1

    #     return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'prostate':
        data_root = args.data_root
        source_csv, valid_csv, target_csv = [], [], []
        sr_tr_img_list = []
        sr_tr_label_list = []

        # --- 0) 基础设置 ---
        DOMAINS = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']

        composed_transforms_tr = A.Compose([
            A.RandomSizedCrop(min_max_height=(300, 330), size=(384, 384), p=0.3),
        ])

        # --- 1) 训练域 CSV（域名 -> *_train.csv；数字表示增强域，后面单独加 PNG） ---
        for id in args.splitid:
            if id in DOMAINS:
                source_csv.append(f'{id}_train.csv')
            else:
                # 数字增强域留到后面用 PNG 加入
                pass

        # --- 2) 验证域 CSV（域名 -> *_val.csv；若没有就兜底 *_train.csv，避免为空） ---
        for id in getattr(args, 'valid', []):
            assert id in DOMAINS, f'prostate valid 只能是域名: got {id}'
            val_path = os.path.join(data_root, f'{id}_val.csv')
            if os.path.exists(val_path):
                valid_csv.append(f'{id}_val.csv')
            else:
                valid_csv.append(f'{id}_train.csv')

        # --- 3) 测试域 CSV（域名 -> *_test.csv；若没有就兜底 *_train.csv） ---
        for id in args.testid:
            assert id in DOMAINS, f'prostate testid 只能是域名: got {id}'
            test_path = os.path.join(data_root, f'{id}_test.csv')
            if os.path.exists(test_path):
                target_csv.append(f'{id}_test.csv')
            else:
                target_csv.append(f'{id}_train.csv')

        # --- 工具：把 convert_labeled_list 的相对路径拼到 data_root 上 ---
        def _abspath_list(root, imgs, msks):
            ai, am = [], []
            for i, m in zip(imgs, msks):
                ii = i if os.path.isabs(i) else os.path.join(root, i)
                mm = m if os.path.isabs(m) else os.path.join(root, m)
                ai.append(ii); am.append(mm)
            return ai, am

        # --- 4) 从 CSV 读训练/验证/测试对 ---
        sr_img_list, sr_label_list   = convert_labeled_list(data_root, source_csv)
        tr_img_list, tr_label_list   = convert_labeled_list(data_root, target_csv)
        val_img_list, val_label_list = convert_labeled_list(data_root, valid_csv) if len(valid_csv) > 0 else ([], [])

        # 路径转绝对
        sr_img_list, sr_label_list     = _abspath_list(data_root, sr_img_list, sr_label_list)
        tr_img_list, tr_label_list     = _abspath_list(data_root, tr_img_list, tr_label_list)
        val_img_list, val_label_list   = _abspath_list(data_root, val_img_list, val_label_list)

        print(f"[INFO] Train CSV: {source_csv}")
        print(f"[INFO] Val   CSV: {valid_csv}")
        print(f"[INFO] Test  CSV: {target_csv}")

        # --- 5) 把“数字增强域”的 PNG 只加入 train ---
        for id in args.splitid:
            if id not in DOMAINS:
                image_dir = os.path.join(data_root, 'train_LAB_LD_prostate', f'Domain{id}', 'image')
                image_list = glob(os.path.join(image_dir, '*.png'))
                sr_tr_img_list += image_list

        # 为增强域构造对应 mask 路径（用 os.sep 更安全）
        for image_path in sr_tr_img_list:
            label_path = image_path.replace(f'{os.sep}image{os.sep}', f'{os.sep}mask{os.sep}')
            sr_tr_label_list.append(label_path)

        # 合并到训练集
        sr_img_list   += sr_tr_img_list
        sr_label_list += sr_tr_label_list

        # --- 6) 可选：存在性过滤（避免后续 Dataset 内再清空） ---
        def _filter_exists(imgs, msks, tag):
            keep_i, keep_m = [], []
            miss = 0
            for i, m in zip(imgs, msks):
                if os.path.exists(i) and os.path.exists(m):
                    keep_i.append(i); keep_m.append(m)
                else:
                    miss += 1
                    if miss <= 8:
                        print(f"[WARN] {tag} 不存在: img={i}  mask={m}")
            if miss > 0:
                print(f"[WARN] {tag} 缺失配对总数: {miss}")
            return keep_i, keep_m

        sr_img_list, sr_label_list   = _filter_exists(sr_img_list, sr_label_list, "TRAIN")
        val_img_list, val_label_list = _filter_exists(val_img_list, val_label_list, "VAL")
        tr_img_list,  tr_label_list  = _filter_exists(tr_img_list,  tr_label_list,  "TEST")

        # --- 7) Debug 头部打印 ---
        # def _head(imgs, msks, tag):
        #     print(f"[DEBUG] {tag}: {len(imgs)} samples")
        #     for k in range(min(3, len(imgs))):
        #         print(f"  {k}: {imgs[k]} | {msks[k]}")
        # _head(sr_img_list,  sr_label_list,  "TRAIN")
        # _head(val_img_list, val_label_list, "VAL")
        # _head(tr_img_list,  tr_label_list,  "TEST")

        # --- 8) DataLoader ---
        train_set = PROSTATE_dataset(
            data_root, sr_img_list, sr_label_list,
            384, args.batch_size, img_normalize=True,
            transform=composed_transforms_tr, aug_wt=args.aug_wt
        )
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            pin_memory=True, num_workers=8, prefetch_factor=1,
            persistent_workers=False, drop_last=True
        )

        # 验证集：不做随机增广；drop_last=False
        val_set = PROSTATE_dataset(
            data_root, val_img_list, val_label_list,
            384, 1, img_normalize=True, transform=None
        )
        val_loader = DataLoader(
            val_set, batch_size=1, shuffle=False,
            pin_memory=False, num_workers=0, drop_last=False
        )

        # 测试集：同上
        test_set = PROSTATE_dataset(
            data_root, tr_img_list, tr_label_list,
            384, 1, img_normalize=True, transform=None
        )
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False,
            pin_memory=False, num_workers=0, drop_last=False
        )
        
        print(f"[INFO] Train set: {len(sr_img_list)} volumes, {len(train_set)} slices")
        print(f"[INFO] Val set:   {len(val_img_list)} volumes, {len(val_set)} slices")
        print(f"[INFO] Test set:  {len(tr_img_list)} volumes, {len(test_set)} slices")

        num_class = 1
        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

