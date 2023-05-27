# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: eval.py
@time: 2023/5/2 16:28
@desc: 
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils import SegmentationMetric
from model import DeepLabV3Plus
from dataset import CamVidDataset
import matplotlib.pyplot as plt


# def denormalize(image, mean, std):
#     for i in range(3):
#         image = [i,:,:] = (image[i, :, :] * std[i]) + mean[i]
#     return image


def evaluate():
    data_dir = r'../dataset/CamVid'  # 数据集所在位置
    model_info_path = r'./checkpoints/deeplabV3Plus_info.pth'  # 训练信息等
    num_class = 32 + 1  # 类别数为32个加1个背景
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.227, 0.226]

    # 加载验证集
    x_valid_dir = os.path.join(data_dir, 'val')
    y_valid_dir = os.path.join(data_dir, 'val_labels')
    valid_dataset = CamVidDataset(x_valid_dir, y_valid_dir)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True
    )
    print("len of valid dataloader: {}".format(len(valid_dataloader)))

    # 加载模型
    model = DeepLabV3Plus(num_class).cuda()
    if os.path.exists(model_info_path) is False:
        print("No train info exist")
        return
    else:
        checkpoints = torch.load(model_info_path)
        model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)  # 该方法是in-place方法
    model.eval()

    # 计算各种指标的混淆矩阵类
    confusionMatrix = SegmentationMetric(num_class=num_class)

    for idx, (images, labels) in enumerate(valid_dataloader):
        with torch.no_grad():
            labels = labels.long()
            images = images.to(device)  # 注意这里Tensor.to()方法不是in-place方法

            output = model(images)
            output = output.cpu()
            confusionMatrix.update(preds=output, labels=labels)

            print("================batch {}================".format(idx))
            print("pixel accuracy: {}".format(confusionMatrix.pixel_accuracy()))
            print("mean IoU: {}".format(confusionMatrix.mean_intersection_over_union()))
            print("FW IoU: {}".format(confusionMatrix.frequency_weighted_intersection_over_union()))

            # 可视化一组图观察一下
            if idx % 10 == 9:
                table = confusionMatrix.summary()
                print(table)
                sample_image = images.cpu()[0]
                sample_predict = output[0]
                sample_label = labels[0]

                sample_predict = F.softmax(sample_predict, dim=0)
                sample_predict = torch.argmax(sample_predict, dim=0)

                plt.figure()
                plt.subplot(131)
                plt.imshow(sample_image.moveaxis(0, 2))
                plt.subplot(132)
                plt.imshow(sample_label, cmap='gray')
                plt.subplot(133)
                plt.imshow(sample_predict, cmap='gray')
                plt.show()
                break
            confusionMatrix.reset()


if __name__ == '__main__':
    evaluate()
