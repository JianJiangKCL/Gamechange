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
from  torch.distributions import multivariate_normal
# from torch.distributions import Normal, MultivariateNormal
# def cov(tensor, rowvar=True, bias=False):
#     """Estimate a covariance matrix (np.cov)
#     https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
#     """
#     tensor = tensor if rowvar else tensor.transpose(-1, -2)
#     tensor = tensor - tensor.mean(dim=-1, keepdim=True)
#     factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
#     return factor * tensor @ tensor.transpose(-1, -2).conj()
# delta = 1e-4
# dim = 512
# n_emb = 512
# x = torch.randn(2000, dim)
# from model.quantizer import Quantizer
# qtz = Quantizer(dim, n_emb)
# x, _, _ = qtz(x)
# best_samples = x
# mean = best_samples.mean(dim=0)
# fs_m = best_samples.sub(mean.expand_as(best_samples))
# cov_mat = fs_m.transpose(0, 1).mm(fs_m) / (len(x) - 1)
# cov_mat = cov_mat + delta * torch.eye(cov_mat.shape[0])
# pd = MultivariateNormal(mean, cov_mat)
# probability = pd.log_prob(x) # torch.exp(pd.log_prob(x))
# print(probability)
# sample = pd.sample()
# probability = pd.log_prob(qtz(x+1)[0])
# print(probability)
#
# normal_dist = MultivariateNormal(torch.zeros(dim)+2, torch.eye(dim))
# test = normal_dist.sample()
# probability = pd.log_prob(test)
# print(probability)
# # samples = pd.sample((self.num_sampling,))
# k=1



