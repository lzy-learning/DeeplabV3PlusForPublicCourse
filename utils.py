# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: utils.py
@time: 2023/5/1 11:31
@desc: 
'''

import torch
import torch.nn.functional as F
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torchnet.meter as meter
import time
import torch
import torchmetrics
from prettytable import PrettyTable


class SegmentationMetric:
    '''提供AP、mAP、IOU、mIOU、FWIOU、precision、recall、specificity
        横着的是真实值，竖着的是预测值'''

    def __init__(self, num_class):
        '''
        :param num_class: 类别数
        '''
        self.num_class = num_class
        self.matrix = np.zeros(shape=(num_class, num_class))
        self.table = PrettyTable()

    def pixel_accuracy(self):
        '''
        PA = (TP+TN) / (TP+TN+FP+FN)
        :return: PA
        '''
        acc = np.diag(self.matrix).sum() / self.matrix.sum()
        return acc

    def class_pixel_accuracy(self):
        '''
        acc = TP / (TP+FP)，即准确率
        :return: ndarray类型的列表，表示每个类别的像素准确率
        '''
        class_acc = np.diag(self.matrix) / self.matrix.sum(axis=1)
        return class_acc

    def summary(self, class_name=None):
        '''
        计算precision, recall, specificity
        '''
        self.table.field_names = ["Category", "Precision", "Recall", "Specificity"]
        if class_name is None:
            class_name = np.arange(self.num_class)
        for i in range(self.num_class):
            TP = self.matrix[i, i]  # 预测正确的在对角线上
            FP = np.sum(self.matrix[i, :]) - TP  # 预测错误的在同一行上
            FN = np.sum(self.matrix[:, i]) - TP  # 假阴性，即属于该类别却没有预测出来
            TN = np.sum(self.matrix) - TP - FP - FN  # 正确预测为不是该类别

            precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0
            recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0
            specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0
            self.table.add_row([class_name[i], precision, recall, specificity])
        return self.table

    def mean_pixel_accuracy(self):
        '''
        平均像素准确率
        :return:
        '''
        class_acc = self.class_pixel_accuracy()
        mean_acc = np.nanmean(class_acc)  # nanmean表示遇到NAN类型当成0处理
        return mean_acc

    def intersection_over_union(self):
        '''
        IOU = TP / (TP+FP+FN)
        :return: IOU
        '''
        intersection = np.diag(self.matrix)
        union = np.sum(self.matrix, axis=1) + np.sum(self.matrix, axis=0) - np.diag(self.matrix)
        iou = intersection / union
        return iou

    def mean_intersection_over_union(self):
        '''
        mIOU = IOU / num_class
        :return: mIOU
        '''
        iou = self.intersection_over_union()
        miou = np.nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self):
        '''
        FWIOU = [(TP+FN)/(TP+FP+TN+FN)]*[TP/(TP+FP+FN)]
        :return: FWIOU
        '''
        freq = np.sum(self.matrix, axis=1) / np.sum(self.matrix)
        iou = self.intersection_over_union()
        FWIOU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIOU

    def generate_confusion_matrix(self, predict, label):
        '''
        根据预测值和真实值生成混淆矩阵
        :param predict: 预测值，注意只有一个样本，而且需要展开为一维
        :param label: 真实值
        :return: 混淆矩阵
        '''
        mask = (label >= 0) & (label < self.num_class)
        # 乘以self.num_class是防止3+2=2+3这种情况
        tmp = self.num_class * label[mask] + predict[mask]
        # bincount是统计array中数的出现的次数，minlength指定结果的长度，否则结果长度为array中最大的那个数
        count = np.bincount(tmp, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        '''
        传入一个batch的预测值和真实值，以更新混淆矩阵
        注意这里传入的是Tensor类型，而且没有经过softmax等操作，需要进行展开等一系列操作
        :param preds: batch*num_class*W*H
        :param label: batch*W*H
        :return: None
        '''
        batch_size = preds.size()[0]

        predictions = F.softmax(preds, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        assert predictions.size() == labels.size()  # 二者的维度必须一致

        tmp_preds = np.array(predictions).flatten()
        tmp_preds = np.reshape(tmp_preds, newshape=(batch_size, -1))
        tmp_labels = np.array(labels).flatten()
        tmp_labels = np.reshape(tmp_labels, newshape=(batch_size, -1))

        for i in range(tmp_preds.shape[0]):
            self.matrix += self.generate_confusion_matrix(tmp_preds[i], tmp_labels[i])

    def reset(self):
        '''
        重置混淆矩阵
        :return:
        '''
        self.matrix = np.zeros((self.num_class, self.num_class))
        self.table = PrettyTable()


class ComputeIoU(torchmetrics.Metric):
    '''计算iou和miou，需要数据已经经过softmax'''

    def __init__(self, num_classes, dist_sync_on_step=False):
        '''
        :param num_classes: 类别数目
        :param dist_sync_on_step: 用于指定在多个进程或设备上运行时，是否在每个训练步骤结束时同步更新度量值
        '''
        super(ComputeIoU, self).__init__(dist_sync_on_step=dist_sync_on_step)
        # dist_reduce_fx指定如何在多个进程或设备之间进行度量值的聚合
        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        '''
        添加数据
        :param preds: 预测结果的概率，需要经过softmax
        :param target: 真实值
        :return:
        '''
        preds = torch.argmax(preds, dim=1)  # 无论是概率还是one-hot形式，都转成索引的形式才方便计算
        for cls in range(self.num_classes):
            pred_c = preds == cls
            target_c = targets == cls
            # 与操作可以筛选出pred和target都为该类别，sum计算总数，(1,2)表示在第1和第2个维度上相加
            intersection = (pred_c & target_c).float().sum((1, 2))
            union = (pred_c | target_c).float().sum((1, 2))
            self.intersection[cls] += intersection.sum()
            self.union[cls] += union.sum()

    def compute(self):
        '''
        :return: miou, iou
        '''
        iou = self.intersection / (self.union + 1e-6)
        miou = iou.mean()
        return miou, iou

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        self.update(preds, targets)
        return self.compute()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)


# 如果要得到平均每次的时间，需要进行封装一下
class TimeMeter(meter.TimeMeter):
    def __init__(self, unit=True):
        super(TimeMeter, self).__init__(unit)
        self.unit = unit
        self.time = time.time()
        self.n = 0

    def add(self, value=None):
        if value is not None:
            self.n += value
        else:
            self.n += 1

    def reset(self):
        self.time = time.time()
        self.n = 0

    def value(self):
        if self.unit:
            return (time.time() - self.time) / self.n
        else:
            return time.time() - self.time


class ConfusionMatrix:
    def __init__(self, num_classes: int, labels: list):
        # 混淆矩阵的一行是预测的一个类别，一列是一个类别的真实值
        self.matrix = np.zeros(shape=(num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        # PrettyTable可以是数据以好看的表格的形式呈现
        self.table = PrettyTable()
        self.accuracy = 0.0

    def update(self, predicts, labels):
        '''往混淆矩阵添加预测结果'''
        for p, l in zip(predicts, labels):
            self.matrix[p, l] += 1

    def summary(self):
        '''打印评价指标'''
        # 计算准确率，即预测正确的比例(包括negative和positive)
        correct_predict = 0
        for i in range(self.num_classes):
            correct_predict += self.matrix[i, i]
        self.accuracy = round(correct_predict / np.sum(self.matrix), 3)  # 保留三位小数

        self.table.field_names = ["Category", "Precision", "Recall", "Specificity"]

        # 对于每个类别，计算它们的准确率PPV、召回率TPR、特异度TNR
        for i in range(self.num_classes):
            TP = self.matrix[i, i]  # 预测正确的在对角线上
            FP = np.sum(self.matrix[i, :]) - TP  # 预测错误的在同一行上
            FN = np.sum(self.matrix[:, i]) - TP  # 假阴性，即属于该类别却没有预测出来
            TN = np.sum(self.matrix) - TP - FP - FN  # 正确预测为不是该类别

            precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0
            recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0
            specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0

            self.table.add_row([self.labels[i], precision, recall, specificity])
        return self.table

    def show(self):
        '''可视化混淆矩阵'''
        matrix = self.matrix

        print("Accuracy: {}%", self.accuracy * 100)
        print(self.table)
        # 从白色到蓝色
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)

        # 在图像旁边显示颜色条
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel("Predicted Labels")
        plt.title("Confusion Matrix")

        # 在图中标注数量/概率信息
        threshold = matrix.max() / 2  # 这个阈值是为了显示颜色更形象
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color='white' if info > threshold else 'black')
        plt.tight_layout()  # 去除子图之间的间隔
        plt.show()


if __name__ == '__main__':
    num_class = 6
    batch_size = 8
    img_size = (224, 224)
    preds = torch.randn(size=(batch_size, num_class, 224, 224))
    labels = torch.randint(low=0, high=num_class, size=(batch_size, 224, 224))
    print("shape of predictions:", preds.size())
    print("shape of labels:", labels.size())

    metric = SegmentationMetric(num_class=num_class)
    metric.update(preds, labels)
    print("PA: {}".format(metric.pixel_accuracy()))
    print("mPA: {}".format(metric.mean_pixel_accuracy()))
    print("iou: {}".format(metric.intersection_over_union()))
    print("miou: {}".format(metric.mean_intersection_over_union()))

    table = metric.summary()
    print(table)
