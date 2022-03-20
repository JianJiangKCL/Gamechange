from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, repeat
from functools import partial, wraps

from quantizer_v3 import Quantizer
from torchvision.models import resnet18, mobilenet_v2

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
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
        self.target = QUANT_TYPE.FEATURE
        self.quantizer = Quantizer(dim, n_emb, decay)
        self.diff = 0
        
    def normal_forward(self, x):
        return x
        
    def quant_forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')
        quantized_x, self.diff, _ = self.quantizer(x)
        quantized_x = rearrange(quantized_x, 'b c (h w) -> b c h w', h=H, w=W)
        return quantized_x

    
class QuantConv_DW(QuantHelper):
    def __init__(self, prod_dw_in_planes, in_planes, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.prod_dw_in_planes = prod_dw_in_planes
        self.in_planes = in_planes
        self.dw_weight = Parameter(torch.empty((in_planes, 1, kernel_size, kernel_size)))
        nn.init.kaiming_normal_(self.dw_weight)
        self.groups = prod_dw_in_planes #* in_planes
        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)
        #print(f'dw inc{self.in_planes}; groups:{self.groups}')

    
    def normal_forward(self, x, *args):
        dw_weight = repeat(self.dw_weight, 'o i h w -> (o repeat) i h w', repeat=self.prod_dw_in_planes//self.in_planes)
        out = self.conv(x, dw_weight)
        self.diff = 0
        return out
    
    def quant_forward(self, x, quantizer):
        reshaped_w = rearrange(self.dw_weight, 'o i h w -> o i (h w)', h=self.kernel_size, w=self.kernel_size)
        quantized_w, self.diff, idxs = quantizer(reshaped_w)
        quantized_w = rearrange(quantized_w, 'o i (h w) -> o i h w', h=self.kernel_size, w=self.kernel_size)
        quantized_w = repeat(quantized_w, 'o i h w -> (o repeat) i h w', repeat=self.prod_dw_in_planes//self.in_planes)
        out = self.conv(x, quantized_w)
        return out
    
    
class QuantConv_PW(QuantHelper):
    def __init__(self, prod_dw_in_planes, in_planes, out_planes, stride=1, padding=0):
        super().__init__()
        self.prod_dw_in_planes = prod_dw_in_planes
        self.in_planes = in_planes

        self.out_planes = out_planes
        self.pw_weight = Parameter(torch.empty((out_planes*in_planes, 1, 1, 1)))
        nn.init.kaiming_normal_(self.pw_weight)
        self.groups = out_planes * prod_dw_in_planes
        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)

        #print(f'pw inc{self.in_planes}; out_c {self.out_planes}; groups:{self.groups}')

    def get_groups(self):
        return self.groups

    def normal_forward(self, x, *args):
        x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.out_planes)
        h, w = self.pw_weight.shape[-2:]
        pw_weight = repeat(self.pw_weight, 'o i h w -> o (repeat i) h w', repeat=self.prod_dw_in_planes//self.in_planes)
        pw_weight = rearrange(pw_weight, 'o i h w -> (o i) () h w', h=h, w=w)
        out = self.conv(x, pw_weight)
        self.diff = 0
        return out
    
    def quant_forward(self, x, quantizer):
        x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.out_planes)
        h, w = self.pw_weight.shape[-2:]
        reshaped_w = rearrange(self.pw_weight, 'o i h w -> o i (h w)')
        quantized_w, self.diff, idxs = quantizer(reshaped_w)

        quantized_w = repeat(quantized_w, 'o i hw -> o (repeat i) hw', repeat=self.prod_dw_in_planes//self.in_planes)
        quantized_w = rearrange(quantized_w, 'o i (h w) -> (o i) () h w', h=h, w=w)
        out = self.conv(x, quantized_w)
        return out


class Quant_dwpw(QuantHelper):
    def __init__(self, prod_dw_in_planes, in_planes, out_planes, n_dw_emb, n_pw_emb, n_f_emb, kernel_size=3, stride=1, padding=0, gs=1, decay=0.99, ret_x=False):
        super().__init__()
        self.ret_x = ret_x
        self.n_f_emb = n_f_emb
        self.decay = decay
        self.prod_dw_in_planes = prod_dw_in_planes
        #print('prod_dw_in_planes:', prod_dw_in_planes)
        self.in_planes = in_planes
        self.dw_conv = QuantConv_DW(prod_dw_in_planes, in_planes, kernel_size, stride, padding)
        self.pw_conv = QuantConv_PW(prod_dw_in_planes, in_planes, out_planes)
        self.groups = self.pw_conv.get_groups()
        self.quantizer_dw = Quantizer(kernel_size ** 2, n_dw_emb)
        self.quantizer_pw = Quantizer(1, n_pw_emb)


    def get_groups(self):
        return self.groups

    def _init_components(self, x):
        b, c, h1, w1 = x.shape
        # self.dw_conv(x, self.quantizer_dw)
        b, c, h2, w2 = self.dw_conv(x, self.quantizer_dw).shape
        self.feat_quantizer_dw = FeatureQuantizer(h1*w1, self.n_f_emb, self.decay).to(x.device)
        self.feat_quantizer_pw = FeatureQuantizer(h2*w2, self.n_f_emb, self.decay).to(x.device)
        
    def normal_forward(self, x):
        h_dw = self.dw_conv(x, self.quantizer_dw)
        h_pw = self.pw_conv(h_dw, self.quantizer_pw)
        return (h_pw, x) if self.ret_x else h_pw
            
    def quant_forward(self, x):
        h_dw = self.feat_quantizer_dw(x)
        h_dw = self.dw_conv(h_dw, self.quantizer_dw)
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
    def __init__(self, n_dw_emb, n_pw_emb, n_f_emb, prod_dw_in_planes, in_planes, out_planes, stride=1, gs=1):
        super().__init__()
        
        self.conv1 = Quant_dwpw(prod_dw_in_planes, in_planes, out_planes, n_dw_emb, n_pw_emb, n_f_emb, stride=stride, padding=1, gs=gs, ret_x=False)

        self.prod_dw_in_planes = self.conv1.get_groups()
        self.conv2 = Quant_dwpw(self.prod_dw_in_planes, out_planes, out_planes*QuantBasicBlock.expansion, n_dw_emb, n_pw_emb, n_f_emb, stride=stride, padding=1, gs=gs, ret_x=False)

        self.prod_dw_in_planes = self.conv2.get_groups()
        self.activation = nn.ReLU(inplace=True)

    def get_prod_dw_in_planes(self):
        return self.prod_dw_in_planes

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        return out
        

class QuantNet(nn.Module):
    def __init__(self, in_channels, n_dw_emb, n_pw_emb, n_f_emb, block, num_blocks, num_classes, gs, out_planes=3):
        super().__init__()
        self.out_planes = out_planes
        self.inplanes = out_planes
        self.prod_dw_in_planes = out_planes
        self.prod_dw_in_planes_list = [self.prod_dw_in_planes]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
        # the first dwpw layer has prod_dw_in_planes = 1
        self.conv2 = self._make_layer(block, out_planes, num_blocks[0], 1, n_dw_emb, n_pw_emb, gs, n_f_emb)
        self.conv3 = self._make_layer(block, out_planes*2, num_blocks[1], 2, n_dw_emb, n_pw_emb, gs, n_f_emb)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(out_planes*2 * block.expansion, num_classes)
    # n_dw_emb, n_pw_emb, n_f_emb, in_planes, out_planes, stride=1, gs=1):
    def _make_layer(self, block, out_planes, num_blocks, stride, n_dw_emb, n_pw_emb, gs, n_f_emb):
        strides = [stride] + [1] * (num_blocks - 1)
        block_net = nn.Sequential()
        for i in range(len(strides)):
            block_net.add_module(f'QuantRes{i+1}', block(n_dw_emb, n_pw_emb, n_f_emb, self.prod_dw_in_planes, self.inplanes, out_planes, strides[i], gs))
            self.inplanes = out_planes * block.expansion
            self.prod_dw_in_planes = block_net[-1].get_prod_dw_in_planes()
            # self.prod_dw_in_planes = self.inplanes * tmp_prod_dw_in_planes * self.prod_dw_in_planes
            #todo the right order of the list
            # self.prod_dw_in_planes_list.append(self.prod_dw_in_planes)
        return block_net
    def _get_groups_pw_out_planes(self):
        group_list = []
        for m in self.modules():
            if isinstance(m, QuantConv_PW):
                group_list.append(m.out_planes)
        return group_list

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        group_list = self._get_groups_pw_out_planes()
        group_list.reverse()
        h = rearrange(h, 'b c h w -> b h w c')
        h = h.view(*h.size()[:3], *group_list, self.out_planes)
        h = h.sum(dim=-1)
        for i in range(len(group_list), 1, -1):
            h = h.sum(dim=i+2)
        h = rearrange(h, 'b h w c-> b c h w')
        # h = rearrange(h, 'b (group c) h w ->  group c b h w ', group=group_list[-1])
        # for i in range(len(group_list) - 1):
        #
        #     h = rearrange(tmp, '(group c) b h w ->  group c b h w ', group=group_list[len(group_list) - 2 - i])
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)



        h = self.head(h)
        return h
    
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