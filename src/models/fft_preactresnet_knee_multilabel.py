"""Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch
from torch._C import _ImperativeEngine
import torch.nn as nn
import torch.nn.functional as F
from .fft_conv import FFTConv2d
from typing import Tuple


def center_crop(data, shape: Tuple[int, int]):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNetFFT_KneeMultilabel(nn.Module):
    def __init__(
        self, block, num_blocks, image_shape, num_classes=2, drop_prob=0.5,
    ):
        super(PreActResNetFFT_KneeMultilabel, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1,
        #                        padding=1, bias=False)

        self.conv1_t2 = FFTConv2d(1, 1, kernel_size=5, stride=1, bias=False)
        self.layernorm = nn.LayerNorm(
            elementwise_affine=False, normalized_shape=[320, 320]
        )

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.dropout = nn.Dropout(p=drop_prob)
        self.image_shape = image_shape

        in_dim = 512 * block.expansion * 100

        # acl, mtear, edema, fracture, cartilage, effusion, cysts

        self.linear_acl = nn.Linear(in_dim, num_classes, bias=False)
        self.linear_mtear = nn.Linear(in_dim, num_classes, bias=False)
        self.linear_edema = nn.Linear(in_dim, num_classes, bias=False)
        self.linear_fracture = nn.Linear(in_dim, num_classes, bias=False)
        self.linear_cartilage = nn.Linear(in_dim, num_classes, bias=False)
        self.linear_effusion = nn.Linear(in_dim, num_classes, bias=False)
        self.linear_cysts = nn.Linear(in_dim, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, kspace):
        kspace = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        kspace = torch.fft.fftn(kspace, dim=(-2, -1))

        out = self.conv1_t2(kspace).abs()

        out = center_crop(out, self.image_shape)

        out = self.layernorm(out)

        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.dropout(out)

        out_acl = self.linear_acl(out)
        out_mtear = self.linear_mtear(out)
        out_edema = self.linear_edema(out)
        out_fracture = self.linear_fracture(out)
        out_cartilage = self.linear_cartilage(out)
        out_effusion = self.linear_effusion(out)
        out_cysts = self.linear_cysts(out)

        return {
            "acl": out_acl,
            "mtear": out_mtear,
            "edema": out_edema,
            "fracture": out_fracture,
            "cartilage": out_cartilage,
            "effusion": out_effusion,
            "cysts": out_cysts,
        }


def PreActResNet18FFT_KneeMultilabel(image_shape, drop_prob=0.5):
    return PreActResNetFFT_KneeMultilabel(
        PreActBlock, [2, 2, 2, 2], drop_prob=drop_prob, image_shape=image_shape
    )


def PreActResNet34FFT_KneeMultilabel(image_shape, drop_prob=0.5):
    return PreActResNetFFT_KneeMultilabel(
        PreActBlock, [3, 4, 6, 3], drop_prob=drop_prob, image_shape=image_shape
    )


def PreActResNet50FFT_KneeMultilabel(image_shape, drop_prob=0.5):
    return PreActResNetFFT_KneeMultilabel(
        PreActBottleneck, [3, 4, 6, 3], drop_prob=drop_prob, image_shape=image_shape
    )


def PreActResNet101FFT_KneeMultilabel(image_shape, drop_prob=0.5):
    return PreActResNetFFT_KneeMultilabel(
        PreActBottleneck, [3, 4, 23, 3], drop_prob=drop_prob, image_shape=image_shape
    )


def PreActResNet152FFT_KneeMultilabel(image_shape, drop_out=0.5):
    return PreActResNetFFT_KneeMultilabel(
        PreActBottleneck, [3, 8, 36, 3], drop_out=drop_out, image_shape=image_shape
    )


def test():
    net = PreActResNet18FFT_KneeMultilabel(drop_out=0.5)
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())


# test()
