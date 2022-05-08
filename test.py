import copy

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, repeat, reduce
from functools import partial, wraps
import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np
import torch
def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.determinstic = True
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Acti(nn.Module):

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(Acti, self).__init__()

    def forward(self, input):
        t = torch.Tensor([0])  # threshold
        out = (input > t).float()
        return out



class PW(nn.Module):
    def __init__(self, prod_dw_in_planes, in_planes, out_planes, group_list):
        super(PW, self).__init__()
        stride = 2
        padding = 1
        self.prod_dw_in_planes = prod_dw_in_planes
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.pw_weight = Parameter(torch.empty((out_planes*in_planes, 1, 1, 1)))
        nn.init.kaiming_normal_(self.pw_weight)
        self.groups = out_planes * prod_dw_in_planes

        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)
        self.sum_conv = partial(F.conv2d, stride=stride, padding=padding)
        self.acti = Acti()
        self.group_list = copy.deepcopy(group_list)
        # self.group_list.reverse()
    def forward(self, x, x2):
        tmp_x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=self.out_planes)
        h, w = self.pw_weight.shape[-2:]
        pw_weight = repeat(self.pw_weight, 'o i h w -> (o repeat) i h w',
                           repeat=self.prod_dw_in_planes // self.in_planes)

        out1 = self.conv(tmp_x, pw_weight)

        tmp_sum_pw_weight = rearrange(self.pw_weight, '(o i) () h w -> o i h w', h=h, w=w, o=self.out_planes, i=self.in_planes)
        out2 = self.sum_conv(x2, tmp_sum_pw_weight)
        relu_out2 = F.relu(out2)

        summed_out1 = reduce(out1, 'b (o i) h w -> b o h w', i=self.in_planes, reduction='sum')
        # the latest layer acti
        flag = acti(summed_out1)
        # repeat validated
        flag = repeat(flag, 'b o h w -> b (o repeat) h w', repeat=self.in_planes)
        out1 = out1 * flag

        relu_summed_out1 = out1
        if self.group_list:
            for in_planes in self.group_list:
                # flag = acti(reduce(relu_summed_out1, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum'))
                # flag = repeat(flag, 'b o h w -> b (o repeat) h w', repeat=in_planes)
                # relu_summed_out1 = relu_summed_out1 * flag
                relu_summed_out1 = reduce(relu_summed_out1, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum')

        # relu_out2 = out2
        is_equal = torch.equal(relu_out2, relu_summed_out1)
        print(is_equal)
        print((relu_out2 - relu_summed_out1).abs().mean())
        # validated relu_out2 and relu_summed_out1 are almost the indentity after the first dwpw
        return out1, relu_summed_out1

class DW(nn.Module):
    def __init__(self, prod_dw_in_planes, in_planes, group_list):
        super(DW, self).__init__()
        stride = 2
        padding = 1
        self.prod_dw_in_planes = prod_dw_in_planes
        self.in_planes = in_planes
        self.dw_weight = Parameter(torch.empty((in_planes, 1, 3, 3)))
        nn.init.kaiming_normal_(self.dw_weight)
        self.groups = prod_dw_in_planes

        self.conv = partial(F.conv2d, stride=stride, padding=padding, groups=self.groups)
        self.sum_conv = partial(F.conv2d, stride=stride, padding=padding,groups=self.in_planes)
        self.group_list = copy.deepcopy(group_list)
        # self.group_list.reverse()
        self.acti = Acti()
        # self.bn = nn.BatchNorm2d(prod_dw_in_planes)
    def forward(self, x, x2):
        dw_weight = repeat(self.dw_weight, 'o i h w -> (o repeat) i h w',
                           repeat=self.prod_dw_in_planes // self.in_planes)
        out1 = self.conv(x, dw_weight)
        # out1 = F.relu(out1)
        
        tmp_sum_dw_weight = self.dw_weight
        out2 = self.sum_conv(x2, tmp_sum_dw_weight)
        # out2 = F.relu(out2)

        summed_out1 = out1
        # for in_planes in self.group_list:
        #     summed_out1 = reduce(summed_out1, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum')

        for in_planes in self.group_list:
            # flag = acti(reduce(summed_out1, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum'))
            # flag = repeat(flag, 'b o h w -> b (o repeat) h w', repeat=in_planes)
            # summed_out1 = summed_out1 * flag
            summed_out1 = reduce(summed_out1, 'b (o i) h w -> b o h w', i=in_planes, reduction='sum')



        is_equal = torch.equal(out2, summed_out1)
        print(is_equal)
        print((out2 - summed_out1).abs().mean())
        return out1, out2

#
in_planes = 3
out_planes = 2
set_seed(1995, 'cuda')
x = torch.randn(1, in_planes, 28, 28)

acti = Acti()
test = acti(x)
group_list = []
prod_dw_in_planes = in_planes

dw1 = DW(prod_dw_in_planes, in_planes, group_list)
group_list.append(in_planes)
pw1 = PW(prod_dw_in_planes, in_planes, out_planes, group_list)
h11, h12 = dw1(x, x)
h21, h22 = pw1(h11, h12)


prod_dw_in_planes = prod_dw_in_planes * out_planes
dw2 = DW(prod_dw_in_planes, out_planes, group_list)
group_list.append(out_planes)
pw2 = PW(prod_dw_in_planes, out_planes, out_planes, group_list)

h31, h32 = dw2(h21, h22)

h41, h42 = pw2(h31, h32)


prod_dw_in_planes = prod_dw_in_planes * out_planes
dw3 = DW(prod_dw_in_planes, out_planes, group_list)
group_list.append(out_planes)
pw3 = PW(prod_dw_in_planes, out_planes, out_planes, group_list)

# h51, h52 = dw3(h41, h42)
# h61, h62 = pw3(h51, h52)
k=1

