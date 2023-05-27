# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: VOC2012.py
@time: 2023/5/21 10:46
@desc: 
'''

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings("ignore")


class VOCDataset(Dataset):
    def __init__(self, image_folder, label_folder, transforms=None):
        super(VOCDataset, self).__init__()
        filenames = os.listdir(image_folder)

        self.image_paths = [os.path.join(image_folder, filename) for filename in filenames]
        self.mask_paths = [os.path.join(label_folder, filename) for filename in filenames]

        delete_indices = [idx for idx in range(len(self.mask_paths)) if not os.path.exists(self.mask_paths[idx])]
        # you must delete from back to front
        for idx in sorted(delete_indices, reverse=True):
            del self.mask_paths[idx]
            del self.image_paths[idx]

        if transforms is None:
            self.transforms = A.Compose([
                A.Resize(224, 224),
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[index]).convert("RGB"))

        pattern = self.transforms(image=image, mask=mask)

        return pattern["image"], pattern["mask"][:, :, 0]
