# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: config.py
@time: 2023/5/20 9:33
@desc: 
'''
import os
import timeit
import datetime
import argparse
import torch

start = timeit.default_timer()
date = datetime.datetime.now()

RESUME = True
RESUME_FILE = ".\\checkpoints\\deeplabv3plus_train_info.pth"
EPOCH = 50
BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
DATA_DIR = "F:\\Datasets"

IGNORE_LABEL = 255
INPUT_SIZE = '224,224'
AUX_LOSS = True
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 32 + 1
RANDOM_SEED = 1024
RESTORE_FROM = r'./checkpoints'
WEIGHT_DECAY = 0.0005
POLY_POWER = 0.09
LOG_DIR = r'./log'


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # select upsampling method
    parser.add_argument('--up-sampling', type=str, default='bilinear',
                        choices=['bilinear', 'nearest', 'bicubic', 'transposed_conv'],
                        help='up sampling method for aspp and decoder')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='fine tuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'camvid': 50
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 8 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'camvid': 0.1
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    return args


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        args = get_args()
        self.args = args
        self.backbone = args.backbone  # 选用的主干网络，可选：['mobilenet', 'resnet', 'drn', 'xception']
        self.output_stride = args.out_stride  # 输出的下采样倍率
        self.up_sampling = args.up_sampling  # aspp和解码器部分使用的上采样方法
        self.dataset = args.dataset  # 数据集名称，可选：['pascal', 'coco', 'cityscapes']
        self.use_sbd = args.use_sbd  # 是否采用微数据集进行预训练
        self.workers = args.workers  # 加载batch时用多少个线程
        self.cuda = not args.no_cuda and torch.cuda.is_available()  # 是否使用GPU加速训练
        self.base_size = args.base_size  # 图片裁剪前的尺寸
        self.crop_size = args.crop_size  # 图片裁剪后的尺寸
        # 使用GPU加速时应指定使用的卡，默认为0，代表使用第0号卡
        if self.cuda:
            try:
                self.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        '''
        传统的归一化需要用到全部数据，每个设备都计算一遍费时费力。通过在每个设备上计算局部均值和方差，然后使用归约操作
        （例如平均值或求和）将它们汇总到全局均值和方差。最后，使用全局均值和方差来进行归一化操作。在有多张卡时可以使用
        '''
        if args.sync_bn is None:
            if self.cuda and len(self.gpu_ids) > 1:
                self.sync_bn = True
            else:
                self.sync_bn = False

        self.freeze_bn = args.freeze_bn  # 是否冻结归一化层的参数，一般来说测试时是要冻结的，训练时则不需要
        self.loss_type = args.loss_type  # 选择哪种损失函数，有交叉熵和focal loss可选

        # 对于不同的数据集使用不同的epoch次数，大的数据集epoch次数少
        if args.epochs is None:
            epochs = {
                'coco': 30,
                'cityscapes': 200,
                'pascal': 50,
            }
            self.epochs = epochs[args.dataset.lower()]

        self.start_epoch = args.start_epoch  # 开始的epoch序号
        # 根据GPU个数去选择batch size
        if args.batch_size is None:
            if self.cuda:
                self.batch_size = 8 * len(args.gpu_ids)
            else:
                self.batch_size = 4
        # 如果不指定测试时的batch size，默认和训练时的一样，不过指定为1对测试更有效
        if args.test_batch_size is None:
            self.test_batch_size = self.batch_size
        # 如果使用平衡权重，每个类别的权重与其在训练数据中的样本数量成反比。样本数量少的类别会被赋予较大的权重。默认不使用
        self.use_balanced_weights = args.use_balanced_weights

        # 学习率的设定
        if args.lr is None:
            lrs = {
                'coco': 0.1,
                'cityscapes': 0.01,
                'pascal': 0.007,
                'camvid': 0.1
            }
            if self.cuda:
                self.lr = lrs[args.dataset.lower()] / (4 * len(self.gpu_ids)) * self.batch_size
            else:
                self.lr = lrs[args.dataset.lower()] / 4 * self.batch_size

        # 学习率衰减策略，默认为poly，可选['poly', 'step', 'cos']
        self.lr_scheduler = args.lr_scheduler
        # 动量参数，默认0.9
        self.momentum = args.momentum
        # 权重衰减参数，默认为5e-4
        self.weight_decay = args.weight_decay

        '''
        Nesterov 方法通过引入一个动量项来更新参数。这个动量项是当前动量和当前位置的负梯度的线性组合。
        通过这种方式，Nesterov 方法在更新参数时综合了当前位置的梯度和之前的动量信息，使得参数更新更具有方向性和稳定性。
        '''
        self.nesterov = args.nesterov  # 默认为False
        self.seed = args.seed  # 随机数种子，默认为1
        self.resume = args.resume  # 默认为恢复文件路径，如果文件不存在则为None
        if args.checkname is None:  # checkpoints检查点文件的名称
            self.checkname = 'deeplab-' + str(self.backbone)

        '''是否先在一个小的数据集上进行训练，以进行参数微调'''
        self.ft = args.ft  # 默认False

        self.eval_interval = args.eval_interval  # 隔多少个epoch进行一次evaluate，默认为1
        self.no_val = args.no_val  # 是否跳过验证环节，默认为False

        torch.manual_seed(self.seed)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                print("=> no checkpoint found at '{}'".format(args.resume))
                self.checkpoint = None
            else:
                self.checkpoint = torch.load(args.resume)
                self.start_epoch = self.checkpoint['epoch']
                args.start_epoch = self.start_epoch
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, self.checkpoint['epoch']))


config = Config()

if __name__ == '__main__':
    print(config.args)
