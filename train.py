# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: train.py
@time: 2023/5/1 10:38
@desc: 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from modeling.deeplab import DeepLab
from dataloaders import make_data_loader
from torch.utils.tensorboard import SummaryWriter
from utils.lr_schedule import LR_Scheduler
from utils.loss import SegmentationLosses
from utils.metric import Evaluator
from utils.saver import Saver
from utils.summary import TensorboardSummary
from tqdm import tqdm
from config import config

class Trainer:
    def __init__(self):
        # 定义保存模型以及各种训练的配置信息的类
        self.saver = Saver()
        self.saver.save_experiment_config() # 保存网络配置信息
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # 准备数据集
        kwargs = {'drop_last':True, 'pin_memory':True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(**kwargs)

        # 定义模型
        self.model = DeepLab(
            num_classes=self.nclass,
            backbone=config.backbone,
            output_stride=config.output_stride,
            sync_bn=config.sync_bn,
            freeze_bn=config.freeze_bn,
            backbone_pretrained=True
        )

        # 定义优化器，网络的各部分学习率不同，主干网络的学习率较低
        optim_params = [{'params': self.model.get_1x_lr_params(), 'lr': config.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': config.lr * 10}]
        self.optimizer = optim.SGD(optim_params, momentum=config.momentum,
                              weight_decay=config.weight_decay, nesterov=config.nesterov)

        # 学习率衰减
        self.scheduler = LR_Scheduler(
            config.lr_scheduler, config.lr,
            config.epochs, len(self.train_loader)
        )
        # 定义损失函数，可选交叉熵或者focol
        self.criterion = SegmentationLosses(weight=None, cuda=config.cuda).build_loss(mode=config.loss_type)

        # 定义评估类
        self.evaluator = Evaluator(self.nclass)

        if config.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=config.gpu_ids)
            self.model = self.model.cuda()

        self.best_pred = 0.0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        i = 0
        image = None

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if config.cuda:
                image, target = image.cuda(), target.cuda()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: {:.3f}'.format(train_loss/(i+1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i+num_img_tr*epoch)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * config.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # 如果没有验证过程就在此保存参数，否则在验证过程保存
        if config.no_val:
            is_best = False
            self.saver.save_checkpoint(
                state={
                    'epoch':epoch+1,
                    'state_dict':self.model.module.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'best_pred':self.best_pred,
                }, is_best=is_best
            )

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\t')
        test_loss = 0.0
        i = 0
        image = None
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if config.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)

            test_loss += loss.item()
            tbar.set_description('Test loss: {:.3f}'.format(test_loss/(i+1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

        acc = self.evaluator.Pixel_Accuracy()
        class_acc = self.evaluator.Pixel_Accuracy_Class()
        miou = self.evaluator.Mean_Intersection_over_Union()
        fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', miou, epoch)
        self.writer.add_scalar('val/Acc', acc, epoch)
        self.writer.add_scalar('val/Acc_class', class_acc, epoch)
        self.writer.add_scalar('val/fwIoU', fwiou, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * config.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(acc, class_acc, miou, fwiou))
        print('Loss: %.3f' % test_loss)

        new_pred = miou
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


if __name__ == '__main__':
    trainer = Trainer()
    print('Starting Epoch:', config.start_epoch)
    print('Total Epoches:', config.epochs)

    for epoch in range(config.start_epoch, config.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)

    trainer.writer.close()