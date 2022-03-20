from enum import Enum

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy


def normalize_tensor(tensor, p=2, dim=0):
    return F.normalize(tensor, p=p, dim=dim)


class QUANT_TYPE(Enum):
    WEIGHT = 1
    FEATURE = 2

    
class Quantizer(nn.Module):
    #decay=0.99 is default value
    def __init__(self, dim, n_embed, decay=0.99, target=QUANT_TYPE.WEIGHT, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.target = target
        self.eps = eps
        self.diff = 0.

        # dxN_e
        embed = torch.randn(dim, n_embed)
        # torch.nn.init.xavier_uniform(embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("acum_embed_sum", torch.zeros_like(self.embed))
        self.register_buffer("acum_embed_onehot_sum", torch.zeros(n_embed))

    def zero_buffer(self):
        self.acum_embed_sum.zero_()
        self.acum_embed_onehot_sum.zero_()

    def update(self):
        # embed_sum_norm = normalize_tensor(self.acum_embed_sum / (self.acum_embed_onehot_sum + self.eps), dim=0)
        embed_sum_norm = self.acum_embed_sum / (self.acum_embed_onehot_sum + self.eps)
        self.embed_avg.mul_(self.decay).add_(embed_sum_norm, alpha=1 - self.decay)
        # self.embed_avg.copy_(embed_sum_norm)
        self.embed.copy_(self.embed_avg)
        
    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = torch.cdist(flatten, self.embed.permute(1, 0), compute_mode="use_mm_for_euclid_dist")
        _, embed_ind = dist.min(1)
        
        # B*H*WxN_e
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1).contiguous() @ embed_onehot
            # print(self.target, flatten.shape, embed_onehot.shape, embed_onehot_sum.shape, embed_sum.shape)
            self.acum_embed_onehot_sum.add_(embed_onehot_sum)
            self.acum_embed_sum.add_(embed_sum)

        diff = (quantize.detach() - input).pow(2).mean()
        self.diff = diff
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1).contiguous())