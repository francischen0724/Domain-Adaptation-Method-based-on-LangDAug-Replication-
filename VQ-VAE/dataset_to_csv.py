import os
import pandas as pd

dataroot = '/provide/your/path'

domains = ['Domain1', 'Domain2', 'Domain3', 'Domain4']

for set in ['train', 'test']:

    for domain in domains:

        image_list = sorted(os.listdir(os.path.join(dataroot, set, domain, 'image')))
        mask_list = sorted(os.listdir(os.path.join(dataroot, set, domain, 'mask')))

        if len(image_list) != len(mask_list):
            print(f'Error: {domain} {set} image and mask count mismatch')
            continue

        for i in range(len(image_list)):
            if os.path.splitext(image_list[i])[0] != os.path.splitext(mask_list[i])[0]:
                print(f'Error: {domain} {set} image and mask mismatch')
                continue

        for i in range(len(image_list)):
            image_list[i] = os.path.join(set, domain, 'image', image_list[i])
            mask_list[i] = os.path.join(set, domain, 'mask', mask_list[i])

        set_pd = pd.DataFrame({'image': image_list, 'mask': mask_list})

        set_pd.to_csv(os.path.join(dataroot, f'{domain}_{set}.csv'), index=False)
        

