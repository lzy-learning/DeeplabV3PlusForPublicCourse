# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: camvid.py
@time: 2023/5/19 13:30
@desc: 
'''

import os
import numpy as np
from mypath import Path
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from utils.mask_encode_decode import decode_segmap
from torchvision import transforms

import warnings

warnings.filterwarnings("ignore")


class CamVidDataset(Dataset):
    def __init__(self, args, split='train'):
        '''you should ensure that the file names of image and mask are the same'''
        super(CamVidDataset, self).__init__()
        self.NUM_CLASS = 32 + 1
        self.args = args
        self.split = split
        if split == 'train':
            image_dir = os.path.join(Path.db_root_dir('camvid'), 'train')
            mask_dir = os.path.join(Path.db_root_dir('camvid'), 'train_labels')
        elif split == 'val':
            image_dir = os.path.join(Path.db_root_dir('camvid'), 'val')
            mask_dir = os.path.join(Path.db_root_dir('camvid'), 'val_labels')
        elif split == 'test':
            image_dir = os.path.join(Path.db_root_dir('camvid'), 'test')
            mask_dir = os.path.join(Path.db_root_dir('camvid'), 'test_labels')
        else:
            raise ValueError
        filenames = os.listdir(image_dir)

        self.image_paths = [os.path.join(image_dir, filename) for filename in filenames]
        self.mask_paths = [os.path.join(mask_dir, filename) for filename in filenames]

        if self.split == 'train':
            self.transforms = A.Compose([
                A.Resize(args.crop_size, args.crop_size),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(args.crop_size, args.crop_size),
                ToTensorV2()
            ])

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[index]).convert("RGB"))
        pattern = self.transforms(image=image, mask=mask)
        sample = {"image": pattern['image'], "label": pattern['mask'][:, :, 0]}
        return sample

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.crop_size = 224
    args.base_size = 224
    dataloader = DataLoader(
        dataset=CamVidDataset(args, 'train'),
        batch_size=8,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=CamVidDataset(args, 'val'),
        batch_size=8,
        shuffle=True,
        drop_last=True
    )

    print("number of batch:", len(dataloader))

    for idx, samples in enumerate(dataloader):
        images = samples['image']
        labels = samples['label']
        print("batch images size:", images.size())
        print("batch labels size:", labels.size())
        plt.figure()
        plt.subplot(221)
        plt.imshow(images[0].moveaxis(0, 2))
        plt.subplot(222)
        rgb_mask = decode_segmap(labels[0].detach().numpy(), dataset='camvid')
        plt.imshow(rgb_mask)
        # plt.imshow(labels[0], cmap='gray')
        break

    for idx, samples in enumerate(test_loader):
        images = samples['image']
        labels = samples['label']
        plt.subplot(223)
        plt.imshow(images[7].moveaxis(0, 2))
        plt.subplot(224)
        rgb_mask = decode_segmap(labels[7].detach().numpy(), dataset='camvid')
        plt.imshow(rgb_mask)
        # plt.imshow(labels[7], cmap='gray')
        plt.show()
        break
