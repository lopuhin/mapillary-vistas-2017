from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import dataset
import utils


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def concat(xs):
    return torch.cat(xs, 1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet(nn.Module):
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels: int=3,
                 filters_base: int=32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 add_output=True):
        super().__init__()
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i]))
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.UpsamplingNearest2d(scale_factor=2)
        upsample_bottom = nn.UpsamplingNearest2d(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], dataset.N_CLASSES, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(concat([x_out, x_skip]))

        if self.add_output:
            x_out = self.conv_final(x_out)
            x_out = F.log_softmax(x_out)
        return x_out


UNet2 = partial(
    UNet,
    down_filter_factors=(1, 2, 4, 8, 16),
    up_filter_factors=(2, 2, 4, 8, 16),
)


class UNet2Scaled(nn.Module):
    output_downscaled = 2
    filters_base = 32
    unet_filters_base = 2 * filters_base
    down_filter_factors = [1, 2, 4, 8]
    up_filter_factors = [2, 2, 4, 8]

    def __init__(self):
        super().__init__()
        b = self.filters_base
        self.head = nn.Sequential(
            Conv3BN(3, b),
            Conv3BN(b, b),
            nn.MaxPool2d(2, 2),
        )
        self.unet = UNet(
            input_channels=b,
            filters_base=self.unet_filters_base,
            up_filter_factors=self.up_filter_factors,
            down_filter_factors=self.down_filter_factors,
            bottom_s=2,
            add_output=False,
        )
        self.output = nn.Sequential(
            Conv3BN(b * 4, b * 4),
            nn.Conv2d(b * 4, dataset.N_CLASSES, 1),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.unet(x)
        x = self.output(x)
        return F.log_softmax(x)


class Loss:
    def __init__(self, dice_weight=0.0, class_weights=None):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            cls_weight = self.dice_weight / dataset.N_CLASSES
            eps = 1e-5
            for cls in range(dataset.N_CLASSES):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss
