from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, repeat, reduce
from functools import partial, wraps


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class QUANT_TYPE(Enum):
    WEIGHT = 1
    FEATURE = 2


class QuantHelper(nn.Module):
    def __init__(self):
        super(QuantHelper, self).__init__()
        self.use_quant = True
        self.target = QUANT_TYPE.WEIGHT
        self.diff = 0
        self.weight_initalized = False

    def set_quant_mode(self, quant=True):
        self.use_quant = quant

    def enable_quant(self):
        self.use_quant = True

    def disable_quant(self):
        self.use_quant = False

    def normal_forward(self, *args):
        pass

    def quant_forward(self, *args):
        pass

    def forward(self, *args):
        if not self.weight_initalized and hasattr(self, '_init_components'):
            self._init_components(*args)
            self.weight_initalized = True
        if self.use_quant:
            return self.quant_forward(*args)
        else:
            return self.normal_forward(*args)


class QuantConv_DW(QuantHelper):
    def __init__(self, in_planes, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.use_quant = False
        self.kernel_size = kernel_size

        self.in_planes = in_planes
        self.dw_weight = Parameter(torch.empty((in_planes, 1, kernel_size, kernel_size)))
        nn.init.kaiming_normal_(self.dw_weight)
        self.groups = in_planes  # * in_planes
        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)

    def forward(self, x, *args):
        out = self.conv(x, self.dw_weight)
        return out


class QuantConv_PW(QuantHelper):
    def __init__(self, in_planes, out_planes, stride=1, padding=0):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pw_weight = Parameter(torch.empty((out_planes * in_planes, 1, 1, 1)))
        nn.init.kaiming_normal_(self.pw_weight)
        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=out_planes*in_planes)
        # self.f_bn = nn.BatchNorm2d(out_planes)
        # print(f'pw inc{self.in_planes}; out_c {self.out_planes}; groups:{self.groups}')
        self.activation = nn.ReLU(inplace=True)


    def forward(self, x, *args):
        x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.out_planes)
        # pw_weight = rearrange(self.pw_weight, '(o i) () h w -> o i h w', o=self.out_planes, i=self.in_planes, h=self.pw_weight.shape[-2], w=self.pw_weight.shape[-1])
        out = self.activation(self.conv(x, self.pw_weight))
        out = reduce(out, 'b (o i) h w -> b o h w', o=self.out_planes, i=self.in_planes, reduction='sum')
        return out




class Quant_dwpw(QuantHelper):
    def __init__(self, in_planes, out_planes,  kernel_size=3, stride=1,
                 padding=0, decay=0.99, ret_x=False):
        super().__init__()
        self.ret_x = ret_x
        self.decay = decay

        # print('prod_dw_in_planes:', prod_dw_in_planes)
        self.in_planes = in_planes
        self.dw_conv = QuantConv_DW(in_planes, kernel_size, stride, padding)
        self.pw_conv = QuantConv_PW(in_planes, out_planes)


        # self.f_bn1 = nn.BatchNorm2d(prod_dw_in_planes)
        # self.f_bn2 = nn.BatchNorm2d(prod_dw_in_planes)



    def forward(self, x, *args):
        dw_out = self.dw_conv(x)
        pw_out = self.pw_conv(dw_out)
        if self.ret_x:
            return pw_out, dw_out
        else:
            return pw_out


class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()

        self.conv1 = Quant_dwpw(in_planes, out_planes, stride=stride, padding=1, ret_x=False)
        self.conv2 = Quant_dwpw(out_planes, out_planes * QuantBasicBlock.expansion,  stride=stride, padding=1, ret_x=False)

        self.activation = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return out


class QuantNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_classes, inplanes=3,  layers=2):
        super().__init__()

        self.inplanes = inplanes
        self.out_planes = inplanes

        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True))
        # the first dwpw layer has prod_dw_in_planes = 1
        self.convs = nn.Sequential()
        strides = [1, 2, 2]
        inplanes_list = [inplanes, inplanes * 2, inplanes * 4]
        self.layers = layers
        for i in range(layers):
            self.convs.add_module(f'dwpw{i + 1}',
                                  self._make_layer(block, inplanes_list[i], num_blocks[i], strides[i]))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.head = nn.Linear(inplanes*2 * block.expansion, num_classes)
        self.head = nn.Linear(inplanes_list[self.layers - 1] * block.expansion, num_classes)


    def _make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        block_net = nn.Sequential()
        for i in range(len(strides)):
            block_net.add_module(f'QuantRes{i + 1}',
                                 block(self.inplanes, out_planes, strides[i]))
            self.inplanes = out_planes * block.expansion

        return block_net

    def forward(self, x):
        h = self.conv1(x)
        h = self.convs(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)

        h = self.head(h)
        return h



def QuantNet9(in_channels, num_classes, **kwargs):
    model = QuantNet(in_channels, QuantBasicBlock, [1, 1], num_classes, **kwargs)
    return model


def QuantNet18(in_channels, num_classes, **kwargs):
    model = QuantNet(in_channels, QuantBasicBlock, [2, 2, 2, 2], num_classes, **kwargs)
    return model