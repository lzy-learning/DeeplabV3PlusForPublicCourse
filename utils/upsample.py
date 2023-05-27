# E:\Program Files(x86)\pyProject\env python
# encoding: utf-8
'''
@author: LZY
@contact: 2635367587@qq.com
@file: upsample.py
@time: 2023/5/26 11:56
@desc: 
'''
import torch
import torch.nn as nn


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels //= self.upscale_factor ** 2
        output_height = height * self.upscale_factor
        output_width = width * self.upscale_factor

        x = x.view(batch_size, channels, self.upscale_factor, self.upscale_factor, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, channels, output_height, output_width)

        return x


if __name__ == '__main__':
    # upscale_factor = 2  # 上采样倍数
    # pixel_shuffle = PixelShuffle(upscale_factor)
    # input_feature = torch.randn(1, 64, 32, 32)  # 输入特征图大小为32x32，通道数为64
    # output_feature = pixel_shuffle(input_feature)  # 进行像素洞填充
    # print(input_feature.size())
    # print(output_feature.size())
    import torch.nn.functional as F

    # 输入特征图
    input_feature = torch.randn(1, 256, 14, 14)  # 示例输入特征图大小为32x32，通道数为64

    # 指定输出图像宽高
    output_height = 56
    output_width = 56

    # 双线性插值
    upscaled_feature_bilinear = F.interpolate(input_feature, size=(output_height, output_width), mode='bilinear')

    # 最近邻插值
    upscaled_feature_nearest = F.interpolate(input_feature, size=(output_height, output_width), mode='nearest')

    # 双三次插值
    upscaled_feature_bicubic = F.interpolate(input_feature, size=(output_height, output_width), mode='bicubic')

    # 反卷积，输出宽高计算公式为：output_width = (input_width-1)*stride-2*padding+kernel_size+output_padding
    kernel_size = 3
    stride = output_height // input_feature.size()[-1]
    padding = 1
    output_padding = output_height - (input_feature.size()[-1]-1)*stride+2*padding-kernel_size
    print("stride", stride)
    print("output padding", output_padding)
    conv_transpose = nn.ConvTranspose2d(
        in_channels=input_feature.size()[1], out_channels=input_feature.size()[1], kernel_size=kernel_size,
        stride=stride, padding=padding, output_padding=output_padding
    )
    upscaled_feature_transpose_conv = conv_transpose(input_feature)

    # 打印上采样后的特征图大小
    print(upscaled_feature_bilinear.size())
    print(upscaled_feature_nearest.size())
    print(upscaled_feature_bicubic.size())
    print(upscaled_feature_transpose_conv.size())

