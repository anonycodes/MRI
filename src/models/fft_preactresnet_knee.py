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
from fastmri.data.transforms import to_tensor
from fastmri.fftc import ifft2c_new


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


class PreActResNetFFT_Knee(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        image_shape,
        data_space, 
        num_classes=2,
        drop_prob=0.5,
        return_features=False,
    ):
        super(PreActResNetFFT_Knee, self).__init__()
        self.in_planes = 64


        self.conv1_t2 = FFTConv2d(1, 1, kernel_size=5, stride=1, bias=False)
        self.layernorm = nn.LayerNorm(
            elementwise_affine=False, normalized_shape=[320, 320]
        )

        self.conv1_all = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_p = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.dropout = nn.Dropout(p=drop_prob)
        self.image_shape = image_shape
        self.data_space = data_space

        in_dim = 512 * block.expansion * 100
        
        self.linear_mtear = nn.Linear(in_dim, num_classes)
        self.linear_acl = nn.Linear(in_dim, num_classes)
        self.linear_abnormal = nn.Linear(in_dim, num_classes)
        self.linear_cartilage = nn.Linear(in_dim, num_classes)
        self.return_features = return_features

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def plot(self, kspace):
        out_complex = kspace[5][0]
        out_mag = out_complex.abs()
        out_phase = out_complex.angle()
        out_mag = center_crop(out_mag, self.image_shape)
        out_complex = center_crop(out_complex, self.image_shape)
        out_phase = center_crop(out_phase, self.image_shape)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10 , 10))
        real = torch.real(out_complex).detach().cpu().numpy()
        imag = torch.imag(out_complex).detach().cpu().numpy()
        ax[0, 0].imshow(real, cmap='gray')
        ax[0,0].set_title('Real')
        ax[0, 1].imshow(imag, cmap='gray')
        ax[0, 1].set_title('Imaginary')

        ax[1, 0].imshow(out_mag.detach().cpu().numpy(), cmap='gray')
        ax[1,0].set_title('Magnitude')
        ax[1,1].imshow(out_phase.detach().cpu().numpy(), cmap='gray')
        ax[1,1].set_title('Phase')
        plt.savefig('test1.png', bbox_inches='tight')


    def forward(self, kspace):
        if self.data_space == "ktoi_w_realimag":
            kspace = kspace.view(kspace.shape[0], kspace.shape[2], kspace.shape[3])
            out = torch.stack((kspace.real, kspace.imag), axis=1).float()
            out = center_crop(out, self.image_shape)
            out = self.conv1_p(out)
        elif self.data_space == "ktoi_w_mag":
            out_mag = kspace.abs()
            out_mag = out_mag.float()
            out_mag = center_crop(out_mag, self.image_shape)
            out = self.conv1(out_mag)
        elif self.data_space == "ktoi_w_magphase":
            kspace = kspace.view(kspace.shape[0], kspace.shape[2], kspace.shape[3])
            out = torch.stack((kspace.abs(), kspace.angle()), axis=1).float()
            out = center_crop(out, self.image_shape)
            out = self.conv1_p(out)
        elif self.data_space == "ktoi_w_all":
            kspace = kspace.view(kspace.shape[0], kspace.shape[2], kspace.shape[3])
            out = torch.stack((kspace.abs(), kspace.angle(), kspace.real, kspace.imag), axis=1).float()
            out = center_crop(out, self.image_shape)
            out = self.conv1_all(out)
        elif self.data_space == "ktoi_w_real":
            out = kspace.real
            out = center_crop(out.float(), self.image_shape)
            out = self.conv1(out)
        elif self.data_space == "ktoi_w_imag":
            out = kspace.imag
            out = center_crop(out.float(), self.image_shape)
            out = self.conv1(out)
        elif self.data_space == "ktoi_w_phase":
            out = kspace.angle()
            out = center_crop(out.float(), self.image_shape)
            out = self.conv1(out)

        out = self.dropout(out)
        layer_1_out = self.layer1(out)
        layer_2_out = self.layer2(layer_1_out)
        layer_3_out = self.layer3(layer_2_out)
        layer_4_out = self.layer4(layer_3_out)
        out = F.avg_pool2d(layer_4_out, 4)
        out = out.view(out.size(0), -1)

        out = self.dropout(out)

        out_abnormal = self.linear_abnormal(out)
        out_mtear = self.linear_mtear(out)
        out_acl = self.linear_acl(out)
        out_cartilage = self.linear_cartilage(out)
        
        if self.return_features:
            return [kspace, layer_1_out, layer_2_out, layer_3_out, layer_4_out]
        else:
            return out_abnormal, out_mtear, out_acl, out_cartilage


def PreActResNet18FFT_Knee(image_shape, data_space, drop_prob=0.5, return_features=False):
    return PreActResNetFFT_Knee(
        PreActBlock, [2, 2, 2, 2], drop_prob=drop_prob, image_shape=image_shape, data_space=data_space, return_features=return_features,
    )


def PreActResNet34FFT_Knee(image_shape, drop_prob=0.5):
    return PreActResNetFFT_Knee(
        PreActBlock, [3, 4, 6, 3], drop_prob=drop_prob, image_shape=image_shape
    )


def PreActResNet50FFT_Knee(image_shape, drop_prob=0.5):
    return PreActResNetFFT_Knee(
        PreActBottleneck, [3, 4, 6, 3], drop_prob=drop_prob, image_shape=image_shape
    )


def PreActResNet101FFT_Knee(image_shape, drop_prob=0.5):
    return PreActResNetFFT_Knee(
        PreActBottleneck, [3, 4, 23, 3], drop_prob=drop_prob, image_shape=image_shape
    )


def PreActResNet152FFT_Knee(image_shape, drop_out=0.5):
    return PreActResNetFFT_Knee(
        PreActBottleneck, [3, 8, 36, 3], drop_out=drop_out, image_shape=image_shape
    )


def test():
    net = PreActResNet18FFT_Knee(drop_out=0.5)
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())

