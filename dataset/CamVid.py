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
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        '''you should ensure that the file names of image and mask are the same'''
        super(CamVidDataset, self).__init__()
        filenames = os.listdir(image_dir)

        self.image_paths = [os.path.join(image_dir, filename) for filename in filenames]
        self.mask_paths = [os.path.join(mask_dir, filename) for filename in filenames]

        self.transforms = A.Compose([
            A.Resize(224, 224),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Normalize(),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[index]).convert("RGB"))

        pattern = self.transforms(image=image, mask=mask)

        return pattern["image"], pattern["mask"][:, :, 0]

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_dir = "F:\\Datasets\\CamVid\\train"
    mask_dir = "F:\\Datasets\\CamVid\\train_labels"

    dataset = CamVidDataset(image_dir, mask_dir)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True
    )

    print("number of data:", len(dataset))
    print("number of batch:", len(dataloader))

    for idx, (images, labels) in enumerate(dataloader):
        print("batch images size:", images.size())
        print("batch labels size:", labels.size())
        plt.figure()
        plt.subplot(221)
        plt.imshow(images[0].moveaxis(0, 2))
        plt.subplot(222)
        plt.imshow(labels[0], cmap='gray')

        plt.subplot(223)
        plt.imshow(images[7].moveaxis(0, 2))
        plt.subplot(224)
        plt.imshow(labels[7], cmap='gray')
        plt.show()
        break