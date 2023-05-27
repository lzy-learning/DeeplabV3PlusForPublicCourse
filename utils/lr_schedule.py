# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: lr_schedule.py
@time: 2023/5/20 9:32
@desc: 
'''
import os
import torch
import torch.optim as optim
import math


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_poly(param_group['lr'], i_iter, total_iters, power)
    return optimizer.param_groups[-1]['lr']


def adjust_learning_rate_by_epoch(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = 0
    if epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9
            lr = p['lr']
    return lr


def adjust_learning_rate_pose(learning_rate, optimizer, epoch):
    decay = 0
    if epoch + 1 >= 10:
        decay = 0.5
    elif epoch + 1 >= 8:
        decay = 0.5
    elif epoch + 1 >= 6:
        decay = 0.5
    elif epoch + 1 >= 4:
        decay = 0.5
    elif epoch + 1 >= 2:
        decay = 0.5
    else:
        decay = 1

    lr = learning_rate * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


if __name__ == '__main__':
    # test the warm-up strategy
    from warmup_scheduler import GradualWarmupScheduler
    import torch.optim as optim
    import matplotlib.pyplot as plt

    params = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer_sgd = optim.SGD(params, lr=0.1)
    lr_schedule = optim.lr_scheduler.StepLR(optimizer_sgd, step_size=50, gamma=1, last_epoch=-1)
    # lr_schedule = optim.lr_scheduler.ExponentialLR(optimizer_sgd, gamma=0.9, last_epoch=-1)
    # 自适应调整学习率的方法
    # lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sgd, mode='min', factor=0.5, patience=5,
    #           threshold=1e-4, threshold_mode='abs', cooldown=0, min_lr=0.001, eps=1e-8)

    schedule_warmup = GradualWarmupScheduler(optimizer_sgd, multiplier=1., total_epoch=5,
                                             after_scheduler=lr_schedule)
    optimizer_sgd.zero_grad()
    optimizer_sgd.step()

    if os.path.exists(r"../checkpoints/test.pth"):
        checkpoints = torch.load(r"../checkpoints/test.pth")
        schedule_warmup.load_state_dict(checkpoints["schedule"])
    epochs = []
    lrs = []
    for epoch in range(1, 20):
        schedule_warmup.step(epoch)
        epochs.append(epoch)
        lrs.append(optimizer_sgd.param_groups[0]['lr'])
        optimizer_sgd.step()

    torch.save({"schedule": schedule_warmup.state_dict()}, r"../checkpoints/test.pth")

    plt.plot(epochs, lrs)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()
