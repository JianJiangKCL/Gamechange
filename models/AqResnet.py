from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, repeat, reduce
from functools import partial, wraps

from models.quantizer import Quantizer
from torchvision.models import resnet18, mobilenet_v2

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(inp, oup, stride=1, groups=1, dilation=1):
    return nn.Conv2d(inp, oup, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


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


class FeatureQuantizer(QuantHelper):
    def __init__(self, dim, n_emb, decay=0.99):
        super().__init__()
        self.use_quant = True
        self.target = QUANT_TYPE.FEATURE
        self.quantizer = Quantizer(dim, n_emb, decay)
        self.diff = torch.Tensor([0]).cuda()

    def normal_forward(self, x):
        return x

    def quant_forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')
        quantized_x, self.diff, _ = self.quantizer(x)
        quantized_x = rearrange(quantized_x, 'b c (h w) -> b c h w', h=H, w=W)
        return quantized_x


class QuantConv_DW(QuantHelper):
    def __init__(self, fake_inplanes, inp, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.use_quant = True
        self.kernel_size = kernel_size
        self.fake_inplanes = fake_inplanes
        self.inp = inp
        self.dw_weight = Parameter(torch.empty((inp, 1, kernel_size, kernel_size)))
        nn.init.kaiming_normal_(self.dw_weight)
        self.groups = fake_inplanes
        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)

    def normal_forward(self, x, *args):
        dw_weight = repeat(self.dw_weight, 'o i h w -> (o repeat) i h w', repeat=self.fake_inplanes // self.inp)
        out = self.conv(x, dw_weight)
        # self.diff = torch.Tensor([0]).cuda()
        return out

    def quant_forward(self, x, quantizer):
        reshaped_w = rearrange(self.dw_weight, 'o i h w -> o i (h w)', h=self.kernel_size, w=self.kernel_size)
        quantized_w, self.diff, _ = quantizer(reshaped_w)
        quantized_w = rearrange(quantized_w, 'o i (h w) -> o i h w', h=self.kernel_size, w=self.kernel_size)
        quantized_w = repeat(quantized_w, 'o i h w -> (o repeat) i h w', repeat=self.fake_inplanes // self.inp)
        out = self.conv(x, quantized_w)

        return out


class QuantConv_PW(QuantHelper):
    def __init__(self, fake_inplanes, inp, oup, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.fake_inplanes = fake_inplanes
        self.inp = inp
        self.use_quant = True
        self.oup = oup
        self.pw_weight = Parameter(torch.empty((oup * inp, 1, kernel_size, kernel_size)))
        nn.init.kaiming_normal_(self.pw_weight)
        self.groups = oup * fake_inplanes
        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)
        # self.f_bn = nn.BatchNorm2d(oup)
        # print(f'pw inc{self.inp}; out_c {self.oup}; groups:{self.groups}')

    def get_groups(self):
        return self.groups

    def normal_forward(self, x, *args):
        x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.oup)
        h, w = self.pw_weight.shape[-2:]
        pw_weight = repeat(self.pw_weight, 'o i h w -> o (repeat i) h w', repeat=self.fake_inplanes // self.inp)
        pw_weight = rearrange(pw_weight, 'o i h w -> (o i) () h w', h=h, w=w)
        # print(f'pw size {pw_weight.shape}')
        out = self.conv(x, pw_weight)

        return out

    def quant_forward(self, x, quantizer):
        x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.oup)
        h, w = self.pw_weight.shape[-2:]
        reshaped_w = rearrange(self.pw_weight, 'o i h w -> o i (h w)')
        quantized_w, self.diff, _ = quantizer(reshaped_w)

        quantized_w = repeat(quantized_w, 'o i hw -> o (repeat i) hw', repeat=self.fake_inplanes // self.inp)
        quantized_w = rearrange(quantized_w, 'o i (h w) -> (o i) () h w', h=h, w=w)
        out = self.conv(x, quantized_w)
        # out = self.f_bn(out)
        return out


# class Quant_IRLB(QuantHelper):
#     '''
#     inverted residual linear bottlenect block
#     '''
#     def __init__(
#             self,
#             inp: int,
#             oup: int,
#             stride: int,
#             expand_ratio: int,
#
#     ) -> None:
#         super(Quant_IRLB, self).__init__()
#         self.stride = stride
#
#         norm_layer = nn.BatchNorm2d
#
#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         layers = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(QuantConv_PW(inp, inp, oup))
#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             norm_layer(oup),
#         ])
#         self.conv = nn.Sequential(*layers)
#         self.out_channels = oup
#         self._is_cn = stride > 1


class Quant_dwpw2(QuantHelper):
    def __init__(self, inp, oup, n_dw_emb, n_pw_emb, n_f_emb, kernel_size=3, stride=1, padding=0, gs=1,
                 decay=0.99, ret_x=False):
        super().__init__()
        self.stride = stride
        self.ret_x = ret_x
        self.n_f_emb = n_f_emb
        self.decay = decay
        self.use_quant = True
        # print('fake_inplanes:', fake_inplanes)
        self.inp = inp
        self.oup = oup
        self.dw_conv = QuantConv_DW(inp, inp, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(inp)
        fake_inplanes = inp
        self.pw_conv = QuantConv_PW(fake_inplanes, inp, oup)

        self.quantizer_dw = Quantizer(kernel_size ** 2, n_dw_emb)
        self.quantizer_pw = Quantizer(1, n_pw_emb)

        self.n_fdw_emb = self.n_f_emb
        self.n_fpw_emb = self.n_f_emb
        ## second block
        fake_inplanes = inp * oup
        self.fake_inplanes = fake_inplanes
        self.dw_conv2 = QuantConv_DW(fake_inplanes, oup, kernel_size, stride, padding)

        self.pw_conv2 = QuantConv_PW(fake_inplanes, oup, oup)
        self.bn2 = nn.BatchNorm2d(oup)
        self.quantizer_dw2 = Quantizer(kernel_size ** 2, n_dw_emb)
        self.quantizer_pw2 = Quantizer(1, n_pw_emb)
        # self.activation = nn.ReLU(inplace=True)
        # leaky relu to make features diverse, have negative values
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.n_fdw_emb2 = self.n_f_emb
        self.n_fpw_emb2 = self.n_f_emb

        self.use_res_connect = self.stride == 1 and inp == oup

    def get_groups(self):
        return self.groups

    def _init_components(self, x):
        b, c, h1, w1 = x.shape
        self.feat_quantizer_dw = FeatureQuantizer(h1 * w1, self.n_fdw_emb, self.decay).to(x.device)

        out = self.dw_conv(x, self.quantizer_dw)
        b, c, h2, w2 = out.shape
        self.feat_quantizer_pw = FeatureQuantizer(h2 * w2, self.n_fpw_emb, self.decay).to(x.device)

        out = self.pw_conv(out, self.quantizer_pw2)
        self.feat_quantizer_dw2 = FeatureQuantizer(h2 * w2, self.n_fdw_emb, self.decay).to(x.device)

        out2 = self.dw_conv2(out, self.quantizer_dw2)
        b, c, h3, w3 = out2.shape
        self.feat_quantizer_pw2 = FeatureQuantizer(h3 * w3, self.n_fpw_emb, self.decay).to(x.device)
        k = 1

    def normal_forward(self, x):
        h = self.dw_conv(x, self.quantizer_dw)
        # h = self.bn1(h)
        h = self.activation(h)

        h = self.pw_conv(h, self.quantizer_pw)

        h = self.dw_conv2(h, self.quantizer_dw2)
        h = self.pw_conv2(h, self.quantizer_pw2)
        h = reduce(h, 'b (gc c) h w -> b gc h w', gc=self.oup, reduction='sum')
        h = self.bn2(h)
        h = self.activation(h)
        return h

    def quant_forward(self, x):
        h_dw = self.feat_quantizer_dw(x)
        h_dw = self.dw_conv(h_dw, self.quantizer_dw)
        # h_dw = self.bn1(h_dw)
        h_dw = self.activation(h_dw)

        h_pw = self.feat_quantizer_pw(h_dw)
        h_pw = self.pw_conv(h_pw, self.quantizer_pw)
        # we don't use activation here, because the maps are not summed up.
        # if activation is used then the summed results will be different from the original results.
        h_pw = self.feat_quantizer_dw2(h_pw)
        h_dw2 = self.dw_conv2(h_pw, self.quantizer_dw2)
        h_pw2 = self.feat_quantizer_pw2(h_dw2)

        h = self.pw_conv2(h_pw2, self.quantizer_pw2)
        # summation
        h = reduce(h, 'b (gc c) h w -> b gc h w', gc=self.oup, reduction='sum')
        h = self.bn2(h)
        h = self.activation(h)
        return h


class Quant_res_pw2(QuantHelper):
    def __init__(self, inp, oup, n_pw_emb, n_f_emb, kernel_size=3, stride=1, padding=0, downsample=None, decay=0.99, ret_x=False):
        super().__init__()
        self.stride = stride
        self.ret_x = ret_x
        self.n_f_emb = n_f_emb
        self.decay = decay
        self.use_quant = True
        # print('fake_inplanes:', fake_inplanes)
        self.inp = inp
        self.oup = oup

        # self.bn1 = nn.BatchNorm2d(inp)
        fake_inplanes = inp
        self.pw_conv = QuantConv_PW(fake_inplanes, inp, oup, kernel_size, stride, padding)

        self.quantizer_pw = Quantizer(kernel_size ** 2, n_pw_emb)
        self.n_fpw_emb = self.n_f_emb

        fake_inplanes = inp * oup
        self.fake_inplanes = fake_inplanes
        # the second conv doesn't have stride as inputs, but we still need padding to keep the shape the same
        self.pw_conv2 = QuantConv_PW(fake_inplanes, oup, oup, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(oup)

        self.quantizer_pw2 = Quantizer(kernel_size ** 2, n_pw_emb)
        self.activation = nn.ReLU(inplace=True)
        # leaky relu to make features diverse, have negative values
        # self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.n_fpw_emb2 = round(self.n_f_emb )
        # *2 cause overload of gpu; /2 make it work
        self.n_fpw_emb_end = round(self.n_f_emb/2)
        # self.use_res_connect = self.stride == 1 and inp == oup

        self.downsample = downsample

    def get_groups(self):
        return self.groups

    def _init_components(self, x):
        h1, w1 = x.shape[-2:]
        self.feat_quantizer_pw = FeatureQuantizer(h1 * w1, self.n_fpw_emb, self.decay).to(x.device)

        out = self.pw_conv(x, self.quantizer_pw)
        h2, w2 = out.shape[-2:]
        self.feat_quantizer_pw2 = FeatureQuantizer(h2 * w2, self.n_fpw_emb2, self.decay).to(x.device)

        out = self.pw_conv2(out, self.quantizer_pw2)
        h_end, w_end = out.shape[-2:]
        self.feat_quantizer_end = FeatureQuantizer(h_end * w_end, self.n_fpw_emb_end, self.decay).to(x.device)

    def normal_forward(self, x):
        identity = x
        h = self.pw_conv(x, self.quantizer_pw)
        # h = self.bn1(h)
        h = self.activation(h)

        h = self.pw_conv2(h, self.quantizer_pw2)

        h = reduce(h, 'b (gc c) h w -> b gc h w', gc=self.oup, reduction='sum')
        h = self.bn2(h)

        if self.downsample is not None:
            identity = self.downsample(x)
        h += identity
        h = self.activation(h)

        return h

    def quant_forward(self, x):
        identity = x
        # print('shape of x ', x.size())
        h_pw = self.feat_quantizer_pw(x)
        h_pw = self.pw_conv(h_pw, self.quantizer_pw)
        # print('size of hpw1', h_pw.size())
        # we don't use activation here, because the maps are not summed up.
        # if activation is used then the summed results will be different from the original results.
        h_pw = self.feat_quantizer_pw2(h_pw)
        h = self.pw_conv2(h_pw, self.quantizer_pw2)
        h = self.feat_quantizer_end(h)
        # print('size of hpw2', h.size())
        # summation
        h = reduce(h, 'b (gc c) h w -> b gc h w', gc=self.oup, reduction='sum')
        h = self.bn2(h)
        # print('after reduce', h.size())
        if self.downsample is not None:
            identity = self.downsample(x)
        h += identity
        h = self.activation(h)
        return h


class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, oup, n_pw_emb, n_f_emb, stride=1, downsample=None, decay=0.99):
        super().__init__()

        self.conv = Quant_res_pw2(inp, oup, n_pw_emb, n_f_emb, stride=stride, padding=1, downsample=downsample, ret_x=False, decay=decay)

    def forward(self, x):
        out = self.conv(x)
        return out

    def straight_mode_(self):
        for m in self.modules():
            if isinstance(m, QuantHelper):
                m.disable_quant()

    def quant_mode_(self):
        for m in self.modules():
            if isinstance(m, QuantHelper):
                m.enable_quant()

    def set_block_qtz_decay(self, decay):
        for m in self.modules():
            if isinstance(m, Quantizer):
                m.set_decay(decay)

    def get_block_weight_codes(self):
        # get layer-wise weight codes; maybe just select the first three, as the last one is not used for fast forward
        codes_dw1 = self.conv.quantizer_dw.codes


class QuantNet(nn.Module):
    def __init__(self, in_channels, n_dw_emb, n_pw_emb, n_f_emb, block, num_blocks, num_classes, gs, oup=3,
                 layers=2):
        super().__init__()
        self.oup = oup

        # inps = [16, 24, 32, 64, 96]  # , 160, 320]
        # inps = [32, 48, 64, 96, 128]  # , 160, 320]
        # inps = [64, 96, 128, 256, 384]
        inps = [16] * 5
        self.inplanes = inps[0]
        n_dw_embs = [n_dw_emb] * 5
        n_pw_embs = [n_pw_emb] * 5
        n_f_embs = [n_f_emb] * 5
        print('inps:', inps)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inps[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inps[0]),
            nn.LeakyReLU(0.1, inplace=True))
        # the first dwpw layer has fake_inplanes = 1
        self.convs = nn.Sequential()
        # if strides=1 will add much more computation
        strides = [1, 2, 2, 2]
        num_blocks = [1, 1, 1, 1]
        # oup_list = [oup, oup * 2, oup * 4]
        # self, block, quant_arch, planes, blocks, stride, n_pw_emb, n_f_emb):
        self.layer1 = self.make_layer(block, inps[0], num_blocks[0], stride=strides[0], n_pw_emb=n_pw_embs[0], n_f_emb=n_f_embs[0])
        self.layer2 = self.make_layer(block, inps[1], num_blocks[1], stride=strides[1], n_pw_emb=n_pw_embs[1], n_f_emb=n_f_embs[1])
        self.layer3 = self.make_layer(block, inps[2], num_blocks[2], stride=strides[2], n_pw_emb=n_pw_embs[2], n_f_emb=n_f_embs[2])
        # self.layers = layers
        # layer_maker = partial(QuantBasicBlock, gs=gs, quant_arch=Quant_res_pw2)
        # decays = [0.99, 0.99]
        #  inp, oup, stride=1,
        # self.layer1 = layer_maker(inp=inps[0], oup=inps[1], stride=strides[0], decay=decays[0], n_dw_emb=n_dw_embs[0],
        #                           n_pw_emb=n_pw_embs[0], n_f_emb=n_f_embs[0])
        #
        # self.layer2 = layer_maker(inp=inps[1], oup=inps[2], stride=strides[1], decay=decays[0], n_dw_emb=n_dw_embs[1],
        #                           n_pw_emb=n_pw_embs[1], n_f_emb=n_f_embs[1])
        #
        # self.layer3 = layer_maker(inp=inps[2], oup=inps[3], stride=strides[2], decay=decays[0], n_dw_emb=n_dw_embs[2],
        #                           n_pw_emb=n_pw_embs[2], n_f_emb=n_f_embs[2])

        # self.layer1.straight_mode_()
        # self.layer2.straight_mode_()
        # self.layer3.straight_mode_()
        # print('straight mode for layer2 and layer3---------')
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Linear(inps[2], num_classes)

    def make_layer(self, block, planes, blocks, stride, n_pw_emb, n_f_emb):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # inp, oup, n_pw_emb, n_f_emb, stride=1, downsample=None, decay=0.99
        layers.append(block(self.inplanes, planes, n_pw_emb=n_pw_emb, n_f_emb=n_f_emb, stride=stride, downsample=downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, n_pw_emb=n_pw_emb, n_f_emb=n_f_emb, stride=stride, downsample=downsample)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv1(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        # h = self.layer4(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)

        h = self.head(h)
        return h

    def set_qtz_decay(self, decay):
        for m in self.modules():
            if isinstance(m, Quantizer):
                m.set_decay(decay)

    def accumlate_diff_feature(self):
        return sum([module.diff for module in self.modules() if
                    hasattr(module, 'target') and module.target == QUANT_TYPE.FEATURE])

    def accumlate_diff_weight(self):
        return sum([module.diff for module in self.modules() if
                    hasattr(module, 'target') and module.target == QUANT_TYPE.WEIGHT])

    def update_quantizer(self):
        for module in self.modules():
            if isinstance(module, Quantizer):
                module.update()
                module.zero_buffer()

    def zero_buffer(self):
        for module in self.modules():
            if isinstance(module, Quantizer):
                module.zero_buffer()

    def disable_quantizer(self):
        for module in self.modules():
            if isinstance(module, QuantHelper):
                module.disable_quant()


def QuantNet9(in_channels, n_dw_emb, n_pw_emb, n_f_emb, num_classes, gs, **kwargs):
    model = QuantNet(in_channels, n_dw_emb, n_pw_emb, n_f_emb, QuantBasicBlock, [1, 1], num_classes, gs, **kwargs)
    return model


def QuantNet18(in_channels, n_dw_emb, n_pw_emb, n_f_emb, num_classes, gs, **kwargs):
    model = QuantNet(in_channels, n_dw_emb, n_pw_emb, n_f_emb, QuantBasicBlock, [2, 2, 2, 2], num_classes, gs, **kwargs)
    return model