import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False, backbone_pretrained=True,
                 up_sampling_='bilinear'):
        super(DeepLab, self).__init__()
        self.up_sampling = up_sampling_
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, backbone_pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, self.up_sampling)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, self.up_sampling)

        if self.up_sampling == 'transposed_conv':
            self.transposed_conv2d = nn.ConvTranspose2d(
                in_channels=num_classes, out_channels=num_classes,
                kernel_size=3, padding=1, stride=4, output_padding=3
            )
        # batch normalization should be frozen during test phase
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        if self.up_sampling == "transposed_conv":
            x = self.transposed_conv2d(x)
        elif self.up_sampling == 'nearest':
            x = F.interpolate(x, size=input.size()[2:], mode=self.up_sampling)
        else:
            x = F.interpolate(x, size=input.size()[2:], mode=self.up_sampling, align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    up_sampling_funcs = ['bilinear', 'nearest', 'bicubic', 'transposed_conv']
    input = torch.rand(1, 3, 224, 224)
    for up_sampling_ in up_sampling_funcs:
        model = DeepLab(
            num_classes=33, backbone='mobilenet', output_stride=16,
            backbone_pretrained=False, up_sampling_=up_sampling_
        )
        model.eval()
        output = model(input)
        print("use {}, and the output size is {}.".format(up_sampling_, output.size()))
