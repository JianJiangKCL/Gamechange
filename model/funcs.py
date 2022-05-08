import torch.nn as nn
import torch

from torch.nn import functional as F
from einops import rearrange

from model.AqResnet import QuantNet9
def create_backbone(args):
    arch = args.arch

    in_channels = None
    num_classes = None
    if args.dataset == 'mnist':
        in_channels = 1
        num_classes = 10
    elif args.dataset == 'cifar10':
        in_channels = 3
        num_classes = 10
    elif args.dataset == 'cifar100':
        in_channels = 3
        num_classes = 100
    if arch == 'resnet9':
        #in_channels, n_dw_emb, n_pw_emb, n_f_emb, num_classes, gs, **kwargs
        backbone = QuantNet9(in_channels, args.n_dw_emb, args.n_pw_emb, args.n_f_emb, num_classes=num_classes, gs=args.gs, out_planes=args.out_planes)
    # elif arch == 'resnet32':
    #     backbone = km_resnet32(n_emb, num_classes=num_classes, gs=args.gs)
    # elif arch == 'resnet50':
    #     backbone = km_resnet50(n_emb, num_classes=num_classes, gs=args.gs)
    else:
        raise NotImplementedError
    return backbone