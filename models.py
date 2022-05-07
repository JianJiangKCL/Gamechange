import copy
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, repeat, reduce
from functools import partial, wraps

# from quantizer_v3 import Quantizer
from classifiers import Quantizer
from torchvision.models import resnet18, mobilenet_v2

class Acti(nn.Module):

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Acti, self).__init__()
        self.t = torch.Tensor([0]).cuda()
    def forward(self, input):

        out = (input > self.t).float()
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class QUANT_TYPE(Enum):
    WEIGHT = 1
    FEATURE = 2

class QuantHelper(nn.Module):
    def __init__(self):
        super(QuantHelper, self).__init__()
        self.use_quant = False
        self.target = QUANT_TYPE.WEIGHT
        self.diff = torch.Tensor([0]).cuda()
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
            # self._init_components(*args)
            self.weight_initalized = True
        if self.use_quant:
            return self.quant_forward(*args)
        else:
            return self.normal_forward(*args)
        

class FeatureQuantizer(QuantHelper):
    def __init__(self, dim, n_emb, decay=0.99):
        super().__init__()
        self.use_quant = False
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
    def __init__(self, prod_dw_in_planes, in_planes, group_list, kernel_size=3, stride=1, padding=0):
        super().__init__()

        self.use_quant = False
        self.kernel_size = kernel_size
        self.prod_dw_in_planes = prod_dw_in_planes
        self.in_planes = in_planes
        self.dw_weight = Parameter(torch.empty((in_planes, 1, 3, 3)))
        nn.init.kaiming_normal_(self.dw_weight)
        self.groups = prod_dw_in_planes

        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)
        self.sum_conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.in_planes)
        self.group_list = copy.deepcopy(group_list)
        self.group_list.reverse()

        self.acti = Acti()

    
    def normal_forward(self, x, x2, *args):
        dw_weight = repeat(self.dw_weight, 'o i h w -> (o repeat) i h w',
                           repeat=self.prod_dw_in_planes // self.in_planes)
        out1 = self.conv(x, dw_weight)
        # out1 = F.relu(out1)

        summed_out1 = out1
        for in_planes in self.group_list:
            summed_out1 = reduce(summed_out1, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum')



        tmp_sum_dw_weight = self.dw_weight
        out2 = self.sum_conv(x2, tmp_sum_dw_weight)

        is_equal = torch.equal(out2, summed_out1)
        print(is_equal)
        print((out2 - summed_out1).abs().mean())
        return out1, out2
    
    def quant_forward(self, x, quantizer):
        reshaped_w = rearrange(self.dw_weight, 'o i h w -> o i (h w)', h=self.kernel_size, w=self.kernel_size)
        quantized_w, self.diff, _ = quantizer(reshaped_w)
        quantized_w = rearrange(quantized_w, 'o i (h w) -> o i h w', h=self.kernel_size, w=self.kernel_size)
        quantized_w = repeat(quantized_w, 'o i h w -> (o repeat) i h w', repeat=self.prod_dw_in_planes//self.in_planes)
        out = self.conv(x, quantized_w)
        # out = self.f_bn(out)
        return out
    
    
class QuantConv_PW(QuantHelper):
    def __init__(self, prod_dw_in_planes, in_planes, out_planes, group_list, stride=1, padding=0):
        super().__init__()
        self.group_list = copy.deepcopy(group_list)
        self.group_list.append(in_planes)
        self.group_list.reverse()
        self.prod_dw_in_planes = prod_dw_in_planes
        self.in_planes = in_planes
        self.use_quant = False
        self.out_planes = out_planes
        self.pw_weight = Parameter(torch.empty((out_planes * in_planes, 1, 1, 1)))
        nn.init.kaiming_normal_(self.pw_weight)
        self.groups = out_planes * prod_dw_in_planes

        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)
        self.sum_conv = partial(F.conv2d, stride=stride, padding=padding)
        self.acti = Acti()

    def get_groups(self):
        return self.groups

    def get_group_list(self):
        return self.group_list

    def normal_forward(self, x, x2, *args):
        tmp_x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.out_planes)
        h, w = self.pw_weight.shape[-2:]
        pw_weight = repeat(self.pw_weight, 'o i h w -> o (repeat i) h w', repeat=self.prod_dw_in_planes // self.in_planes)
        pw_weight = rearrange(pw_weight, 'o i h w -> (o i) () h w', h=h, w=w)
        out1 = self.conv(tmp_x, pw_weight)

        summed_out1 = reduce(out1, 'b (o i) h w -> b o h w', i=self.in_planes, reduction='sum')

        # the latest layer acti
        flag = self.acti(summed_out1)
        flag = repeat(flag, 'b o h w -> b (o repeat) h w', repeat=self.in_planes)
        out1 = out1 * flag

        relu_summed_out1 = out1
        if self.group_list:
            for in_planes in self.group_list:
                relu_summed_out1 = reduce(relu_summed_out1, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum')

        tmp_sum_pw_weight = rearrange(self.pw_weight, '(o i) () h w -> o i h w', h=h, w=w, o=self.out_planes,
                                      i=self.in_planes)
        out2 = self.sum_conv(x2, tmp_sum_pw_weight)
        relu_out2 = F.relu(out2)
        is_equal = torch.equal(relu_out2, relu_summed_out1)
        print(is_equal)
        print((relu_out2 - relu_summed_out1).abs().mean())
        return out1, relu_out2
    
    def quant_forward(self, x, quantizer):
        x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.out_planes)
        h, w = self.pw_weight.shape[-2:]
        reshaped_w = rearrange(self.pw_weight, 'o i h w -> o i (h w)')
        quantized_w, self.diff, _ = quantizer(reshaped_w)

        quantized_w = repeat(quantized_w, 'o i hw -> o (repeat i) hw', repeat=self.prod_dw_in_planes//self.in_planes)
        quantized_w = rearrange(quantized_w, 'o i (h w) -> (o i) () h w', h=h, w=w)
        out = self.conv(x, quantized_w)
        # out = self.f_bn(out)
        return out


class Quant_dwpw(QuantHelper):
    def __init__(self, prod_dw_in_planes, in_planes, out_planes, group_list, n_dw_emb, n_pw_emb, n_f_emb, kernel_size=3, stride=1, padding=0, gs=1, decay=0.99, ret_x=False):
        super().__init__()
        self.ret_x = ret_x
        self.n_f_emb = n_f_emb
        self.decay = decay
        self.prod_dw_in_planes = prod_dw_in_planes
        #print('prod_dw_in_planes:', prod_dw_in_planes)
        self.in_planes = in_planes
        self.dw_conv = QuantConv_DW(prod_dw_in_planes, in_planes, group_list, kernel_size, stride, padding)
        self.pw_conv = QuantConv_PW(prod_dw_in_planes, in_planes, out_planes, group_list)
        self.groups = self.pw_conv.get_groups()

        self.quantizer_dw = Quantizer(kernel_size ** 2, n_dw_emb)
        self.quantizer_pw = Quantizer(1, n_pw_emb)
        self.n_fdw_emb = self.n_f_emb * prod_dw_in_planes
        self.n_fpw_emb = self.n_f_emb * out_planes
        # self.f_bn1 = nn.BatchNorm2d(prod_dw_in_planes)
        # self.f_bn2 = nn.BatchNorm2d(prod_dw_in_planes)

    def get_groups(self):
        return self.groups

    def get_group_list(self):
        return self.pw_conv.get_group_list()

    def _init_components(self, x):
        b, c, h1, w1 = x.shape
        # self.dw_conv(x, self.quantizer_dw)
        b, c, h2, w2 = self.dw_conv(x, self.quantizer_dw).shape
        self.feat_quantizer_dw = FeatureQuantizer(h1*w1, self.n_fdw_emb, self.decay).to(x.device)
        self.feat_quantizer_pw = FeatureQuantizer(h2*w2, self.n_fpw_emb, self.decay).to(x.device)
        
    def normal_forward(self, x, x2):
        h_dw, h_dw2 = self.dw_conv(x, x2)
        h_pw, h_pw2 = self.pw_conv(h_dw, h_dw2, self.quantizer_pw)
        return (h_pw, x) if self.ret_x else h_pw, h_pw2
            
    def quant_forward(self, x):
        # h_dw = self.f_bn1(x)
        h_dw = self.feat_quantizer_dw(x)

        h_dw = self.dw_conv(h_dw, self.quantizer_dw)
        # h_dw = self.f_bn2(h_dw)
        h_pw = self.feat_quantizer_pw(h_dw)

        h_pw = self.pw_conv(h_pw, self.quantizer_pw)
        return (h_pw, x) if self.ret_x else h_pw
    
# class Last_FC(nn.Module):
#     def __init__(self, in_planes, out_planes, decay=0.99):
#         super().__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.fc = nn.Linear(in_planes, out_planes)
#         self.feat_quantizer = FeatureQuantizer(out_planes, n_f_emb, decay).to(x.device)
class QuantBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, n_dw_emb, n_pw_emb, n_f_emb, prod_dw_in_planes, in_planes, out_planes, group_list, stride=1, gs=1):
        super().__init__()
        
        self.conv1 = Quant_dwpw(prod_dw_in_planes, in_planes, out_planes, group_list, n_dw_emb, n_pw_emb, n_f_emb, stride=stride, padding=1, gs=gs, ret_x=False)

        self.prod_dw_in_planes = self.conv1.get_groups()
        group_list = self.conv1.get_group_list()
        self.conv2 = Quant_dwpw(self.prod_dw_in_planes, out_planes, out_planes*QuantBasicBlock.expansion, group_list, n_dw_emb, n_pw_emb, n_f_emb, stride=stride, padding=1, gs=gs, ret_x=False)
        self.group_list = self.conv2.get_group_list()
        self.prod_dw_in_planes = self.conv2.get_groups()
        self.activation = nn.ReLU(inplace=True)

    def get_group_list(self):
        return self.group_list
    def get_prod_dw_in_planes(self):
        return self.prod_dw_in_planes

    def forward(self, x, x2):
        out, out2 = self.conv1(x, x2)
        # out2 = self.activation(out2)
        out, out2 = self.conv2(out, out2)
        # out2 = self.activation(out2)
        return out, out2
        

class QuantNet(nn.Module):
    def __init__(self, in_channels, n_dw_emb, n_pw_emb, n_f_emb, block, num_blocks, num_classes, gs, inplanes=3, layers=2):
        super().__init__()
        self.tmp_group_list = []
        self.inplanes = inplanes
        self.out_planes = inplanes
        self.prod_dw_in_planes = inplanes
        self.prod_dw_in_planes_list = [self.prod_dw_in_planes]
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(in_channels, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True))
        # the first dwpw layer has prod_dw_in_planes = 1
        self.convs = nn.Sequential()
        strides = [1, 2, 2]
        inplanes_list = [inplanes, inplanes*2, inplanes*4]
        self.layers = layers
        for i in range(layers):
            self.convs.add_module(f'dwpw{i+1}', self._make_layer(block, inplanes_list[i], num_blocks[i], strides[i], n_dw_emb, n_pw_emb, gs, n_f_emb))

        # self.conv2 = self._make_layer(block, inplanes, num_blocks[0], 1, n_dw_emb, n_pw_emb, gs, n_f_emb)
        # self.conv3 = self._make_layer(block, inplanes*2, num_blocks[1], 2, n_dw_emb, n_pw_emb, gs, n_f_emb)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.head = nn.Linear(inplanes*2 * block.expansion, num_classes)
        self.head = nn.Linear(inplanes_list[self.layers-1] * block.expansion, num_classes)
        self.activation = nn.ReLU(inplace=True)
        self._get_groups_pw_out_planes()
    # n_dw_emb, n_pw_emb, n_f_emb, in_planes, out_planes, stride=1, gs=1):
    def _make_layer(self, block, out_planes, num_blocks, stride, n_dw_emb, n_pw_emb, gs, n_f_emb):
        strides = [stride] + [1] * (num_blocks - 1)
        block_net = nn.Sequential()
        for i in range(len(strides)):

            block_net.add_module(f'QuantRes{i+1}', block(n_dw_emb, n_pw_emb, n_f_emb, self.prod_dw_in_planes, self.inplanes, out_planes, self.tmp_group_list, strides[i], gs))
            self.inplanes = out_planes * block.expansion
            self.prod_dw_in_planes = block_net[-1].get_prod_dw_in_planes()
            self.tmp_group_list = block_net[-1].get_group_list()

        return block_net

    def _get_groups_pw_out_planes(self):
        group_list = []
        for m in self.modules():
            if isinstance(m, QuantConv_PW):
                group_list.append(m.out_planes)
        group_list.reverse()
        self.group_list = group_list

    def forward(self, x):
        h = self.conv1(x)
        h, h2 = self.convs(h, h)

        summed_h = h
        for in_planes in self.group_list:
            summed_h = F.relu(reduce(summed_h, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum'))

        h = self.avg_pool(summed_h)
        h = h.view(h.size(0), -1)
        pred = self.head(h)
        
        h2 = self.avg_pool(h2)
        h2 = h2.view(h2.size(0), -1)

        pred2 = self.head(h2)
        
        return pred, pred2

    
    def accumlate_diff_feature(self):
        return sum([module.diff for module in self.modules() if hasattr(module, 'target') and module.target == QUANT_TYPE.FEATURE])
    
    def accumlate_diff_weight(self):
        return sum([module.diff for module in self.modules() if hasattr(module, 'target') and module.target == QUANT_TYPE.WEIGHT])
    
    def update_quantizer(self):
        for module in self.modules():
            if isinstance(module, Quantizer):
                module.update()
                module.zero_buffer()
                
    def zero_buffer(self):
        for module in self.modules():
            if isinstance(module, Quantizer):
                module.zero_buffer()
    
    
def QuantNet9(in_channels, n_dw_emb, n_pw_emb, n_f_emb, num_classes, gs, **kwargs):
    model = QuantNet(in_channels, n_dw_emb, n_pw_emb, n_f_emb, QuantBasicBlock, [1, 1], num_classes, gs, **kwargs)
    return model

def QuantNet18(in_channels, n_dw_emb, n_pw_emb, n_f_emb,  num_classes, gs, **kwargs):
    model = QuantNet(in_channels, n_dw_emb, n_pw_emb, n_f_emb, QuantBasicBlock, [2, 2, 2, 2], num_classes, gs, **kwargs)
    return model