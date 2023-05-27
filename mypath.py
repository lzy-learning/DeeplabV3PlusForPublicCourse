# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: mypath.py
@time: 2023/5/25 23:43
@desc: 
'''
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'camvid':
            return "D:\\Datasets\\CamVid"
        else:
            print('dataset {} not available.'.format(dataset))
            raise NotImplementedError
