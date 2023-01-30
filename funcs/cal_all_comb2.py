import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from models.AqResnet import  QuantConv_DW,FeatureQuantizer, QuantConv_PW, QuantNet9, Quant_dwpw2, QuantBasicBlock
from torchvision.datasets import MNIST
from tqdm import tqdm
import os
import argparse
import json
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from utils import *
from einops import rearrange, repeat
from numpy import sqrt


def dw_comb(in_qtz, w_qtz, out_qtz, layer):
    # we directly have 1 to 1 mapping (and get one output of out_emb) between in_embs and w_embs
    # w_qtz
    # w_embs
    # w_codes_dict_list; the bought_codes in forward_debug is exatcly the same as the our w_codes in dict
    # layer stride and padding
    # in_test_emb
    # in_qtz
    # w_test_emb; the right order is selected
    comb_keys = {}
    h_in = w_in = int(sqrt(in_qtz.embed.shape[0]))
    n_w = w_qtz.embed.size(1)
    n_in = in_qtz.embed.size(1)

    in_embs = rearrange(in_qtz.embed, '(H W) n_f-> n_f H W', H=h_in, W=w_in)
    # the reason why whe repeat the in_embs is that we want to get the same shape as the w_embs,
    # so that every combination can be calculated.
    repeated_in_embs = repeat(in_embs, 'n_f H W -> (repeat) n_f  H W', repeat=n_w)
    repeated_in_embs = rearrange(repeated_in_embs, 'n_w n_f  H W -> () (n_w n_f) H W')

    h_w = int(sqrt(w_qtz.embed.shape[0]))
    w_w = int(sqrt(w_qtz.embed.shape[0]))
    w_embs = rearrange(w_qtz.embed, '(h w) n_w-> n_w h w', h=h_w, w=w_w)

    # in_test_embs = in_embs[torch.LongTensor([14, 22, 15])].unsqueeze(0) # exactly the same as corresponding debug foward
    # # w_test_embs_all = w_embs[w_codes_all]
    # w_test_embs = w_embs[torch.LongTensor([3, 49, 17])].unsqueeze(1)
    # test_out = F.conv2d(in_test_embs, w_test_embs, None, layer.stride, layer.padding, groups=3)
    # test_out = rearrange(test_out, 'b c h w -> b c (h w)')
    # test_qtz_out = out_qtz.encode(test_out)

    repeated_w_embs = repeat(w_embs, 'n_w h w ->  n_w (repeat)  h w', repeat=n_in)
    repeated_w_embs = rearrange(repeated_w_embs, 'n_w n_f h w -> (n_w n_f) () h w')
    out = F.conv2d(repeated_in_embs, repeated_w_embs, None, layer.stride, layer.padding, groups=n_in * n_w)
    out = rearrange(out, 'b c h w -> b c (h w)')
    out_indices = out_qtz.encode(out)
    out_indices = rearrange(out_indices, '() (n_w n_f) -> n_w n_f ', n_f=n_in, n_w=n_w).detach().cpu()
    for w_id, out_ind in enumerate(out_indices):
        for f_id, out_ind_f in enumerate(out_ind):
            key = (f_id, w_id)
            comb_keys[key] = out_ind_f.item()
    # test_out14_43 =comb_keys[(14, 43)] # 30
    # test_out22_45 =comb_keys[(22, 45)] #26
    # test_out_17_17 = comb_keys[(15, 17)] # 26
    return comb_keys


# for dw_group; two-level fake quantization is required
def get_dw_group_embs(in_qtz, w_qtz, layer):
    ## validated
    # in_qtz
    # layer
    # w_qtz
    grouped_channel_number = 2
    comb_keys = {}
    h_in = int(sqrt(in_qtz.embed.shape[0]))
    w_in = int(sqrt(in_qtz.embed.shape[0]))
    n_w = w_qtz.embed.size(1)
    n_in = in_qtz.embed.size(1)

    in_embs = rearrange(in_qtz.embed, '(H W) n-> n H W', H=h_in, W=w_in)
    repeated_in_embs = repeat(in_embs, ' n_f H W -> (repeat) n_f H W', repeat=n_w)
    repeated_in_embs = rearrange(repeated_in_embs, 'n_w n_f H W -> () (n_w n_f) H W')

    h_w = int(sqrt(w_qtz.embed.shape[0]))
    w_w = int(sqrt(w_qtz.embed.shape[0]))
    w_embs = rearrange(w_qtz.embed, '(h w) n_w-> n_w h w', h=h_w, w=w_w)
    repeated_w_embs = repeat(w_embs, 'n_w h w ->   n_w (repeat) h w', repeat=n_in)
    repeated_w_embs = rearrange(repeated_w_embs, 'n_w n_f h w -> (n_w n_f) () h w')
    # all possible combination of fq with dw_group
    out = F.conv2d(repeated_in_embs, repeated_w_embs, None, stride=layer.stride, padding=layer.padding, groups=n_in * n_w)
    # we can regard the fake_id as temporary id. as the output is not directly quantized by the next quantizer
    # [ n_f * n_w, h , w]
    out = rearrange(out, '() n h w -> n h w')
    out.n_f = len(in_embs)
    out.n_w = len(w_embs)
    fake_dw_group_comb_keys = {}
    fake_key2comb = []
    fake_fq = rearrange(out, '(n_w n_f) h w -> n_w n_f h w', n_f=n_in, n_w=n_w).detach().cpu()
    fake_id = 0
    for w_id, out_ind in enumerate(fake_fq):
        for f_id, _ in enumerate(out_ind):
            key = (f_id, w_id)
            fake_dw_group_comb_keys[key] = fake_id
            fake_key2comb.append(key)
            fake_id += 1

    return fake_dw_group_comb_keys, out#, cartesian_indices


def get_dwpw_comb(dwpw):
    wq_dw = dwpw.qtz_dw
    wq_pw = dwpw.qtz_pw
    fq_dw = dwpw.fq_dw.quantizer
    fq_dw2 = dwpw.fq_dw2.quantizer
    # get layer to pass its stride and padding
    dw = dwpw.dw
    pw = dwpw.dw_group
    dw_comb_keys = dw_comb(fq_dw, wq_dw, fq_dw2, dw)

    fake_dw_group_comb_keys, pw_embs = get_dw_group_embs(fq_dw2, wq_pw, pw)
    return dw_comb_keys, fake_dw_group_comb_keys, pw_embs

import torch.utils.data as data
class RuntimeEmbDataset(data.Dataset):
    def __init__(self, embs, ret_fake_comb=False, ret_ids=False):
        #[n_f * n_w, 2]; because this is gc=2 group convolution
        self.embs = embs
        # self.cartesian_indices = cartesian_indices
        self.ret_fake_comb = ret_fake_comb
        self.ret_ids = ret_ids
        # self.fake_comb_keys = {}
    def __getitem__(self, index):
        # ids = self.cartesian_indices[index]

        fake1_id = int(index / len(self.embs))
        fake2_id = int(index % len(self.embs))
        ids = (fake1_id, fake2_id)
        # if self.ret_fake_comb and ids not in self.fake_comb_keys:
        #     self.fake_comb_keys[ids] = index

        ids = torch.LongTensor([fake1_id, fake2_id])
        embs = self.embs[ids]
        # out = embs.sum(dim=0)

        return embs, ids
    def __len__(self):
        return len(self.embs)**2

# within a block, first kmconv to the second kmconv
# no fake quantization is required here
def get_comb_consecutive_layer(next_fqtz, dw_group_embs):
    batch_size = 2048
    ds = RuntimeEmbDataset(dw_group_embs.cpu(), ret_fake_comb=True, ret_ids=False)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
    all_comb_indices = []
    next_fqtz = next_fqtz.cuda()
    loader = tqdm(loader)
    feature_com_keys = {}
    fake_comb_indices_total = []
    comb_indices_total = []
    for embs,  fake_comb_indices in loader:
        embs = embs.cuda()
        summed_embs = embs.sum(dim=1)
        summed_embs = summed_embs
        summed_embs = F.relu(summed_embs)
        summed_embs = rearrange(summed_embs, 'b h w -> b (h w)')
        comb_indices = next_fqtz.encode(summed_embs).detach().cpu()
        fake_comb_indices_total.append(fake_comb_indices)
        comb_indices_total.append(comb_indices)
        # if do the for loop inside, it will be 10 times slower
        # for k, v in zip(fake_comb_indices, comb_indices):
        #     k = tuple(k.tolist())
        #     feature_com_keys[k] = v.item()
    print('encoding finishes')
    fake_comb_indices_total = torch.cat(fake_comb_indices_total, dim=0)
    # fake_comb_indices_total = fake_comb_indices_total.flatten()
    comb_indices_total = torch.cat(comb_indices_total, dim=0)
    print('concatenating finishes')
    # comb_indices_total = comb_indices_total.flatten()
    for k, v in zip(fake_comb_indices_total, comb_indices_total):
        k = tuple(k.tolist())
        feature_com_keys[k] = v.item()
    return feature_com_keys


def get_comb_for_Block(block, next_block_first_f_qtz):
    dwpw_list = []
    shortcut = None
    has_shortcut = False
    # if block.stride != 1 or block.in_channels != KmBasicBlock.expansion * block.out_channels:
    #     has_shortcut = True
    for name, m in block.named_modules():
        if isinstance(m, Quant_dwpw2):
            dwpw_list.append(m)

        # if isinstance(m, KM1x1_shortcut_pw_as_dw):
        #     if has_shortcut:
        #         shortcut = m
    dwpw1, dwpw2 = dwpw_list[0], dwpw_list[1]
    # for dwpw1
    dw_comb_keys1, fake_dw_group_comb_keys1, pw_embs1 = get_dwpw_comb(dwpw1)
    feature_com_keys1 = get_comb_consecutive_layer(dwpw2.fq_dw.quantizer, pw_embs1)
    # for dwpw2
    dw_comb_keys2, fake_dw_group_comb_keys2, pw_embs2 = get_dwpw_comb(dwpw2)

    feature_com_keys2 = get_comb_consecutive_layer(next_block_first_f_qtz, pw_embs2)
    return dw_comb_keys1, fake_dw_group_comb_keys1, feature_com_keys1, dw_comb_keys2, fake_dw_group_comb_keys2, None, None, None, feature_com_keys2


def get_comb_for_model(model):
    blocks = []
    for name, m in model.named_modules():
        if isinstance(m, KmBasicBlock):
            blocks.append(m)
    next_block_first_f_qtz = None
    comp_keys_list = []
    for i, block in enumerate(blocks):
        if i < len(blocks) - 1:
            next_block = blocks[i + 1]
            for name, m in next_block.named_modules():
                if isinstance(m, Feature_Quantizer):
                    next_block_first_f_qtz = m.quantizer
                    break

        else:  # last block
            next_block_first_f_qtz = model.last_fqtz.quantizer
        print(f'{i} block keys building finished')

        dw_comb_keys1, fake_dw_group_comb_keys1, feature_com_keys1, dw_comb_keys2, fake_dw_group_comb_keys2, fake_sc_comb_keys1, fake_group_com_keys_add, fake_sc_comb_keys_add, feature_com_keys2 = get_comb_for_Block(
            block, next_block_first_f_qtz)

        comb_dict = {'dw1': dw_comb_keys1, 'dw1_fake': fake_dw_group_comb_keys1, 'dw2': dw_comb_keys2,
                     'dw2_fake': fake_dw_group_comb_keys2, 'feature1': feature_com_keys1, 'feature2': feature_com_keys2,
                     'sc1': fake_sc_comb_keys1, 'sc2': fake_sc_comb_keys_add, 'group_add': fake_group_com_keys_add}
        comp_keys_list.append(comb_dict)
    return comp_keys_list


def fast_forward_dw(f_codes, w_codes, comb_keys):
    #  f_codes: [b, in_c]
    #  w_codes: [in_c, 1 ]
    # out_codes: [b, in_c]
    batch_size = f_codes.shape[0]
    f_codes = rearrange(f_codes, 'b c -> (b c)')
    repeated_w_codes = repeat(w_codes, 'c ()-> (repeat c)', repeat=batch_size)

    out_codes = comb_keys[f_codes, repeated_w_codes]
    out_codes = rearrange(out_codes, '(b c) -> b c', b=batch_size)
    # out_codes = []
    # for f_code, w_code in zip(f_codes, repeated_w_codes):
    #     f_code = f_code.item()
    #     w_code = w_code.item()
    #     out_code = comb_keys[(f_code, w_code)]
    #     out_codes.append(torch.LongTensor([out_code]))
    # out_codes = torch.stack(out_codes, dim=0)
    # out_codes = rearrange(out_codes, '(b c) () -> b c', b=batch_size)
    return out_codes, f_codes


def fast_forward_dw_group(f_codes, w_codes, fake_comb_keys, feature_com_keys):
    # only return fake_comb_keys
    #  f_codes: [b, gc*gs=in_c]
    #  w_codes: [gs*k=out_c, in_c//gs=gc=2]
    # out_codes: [b, out_c]
    batch_size = f_codes.shape[0]
    out_channel = w_codes.shape[0]
    gs = int(f_codes.shape[1] // w_codes.shape[1])
    # number of concatenated kernels in each group of w_codes
    k = int(w_codes.shape[0] // gs)
    # w_codes is [out_c, in_c// gs]
    # w_codes doesn't need to be repeated; f_codes needs repeated
    f_codes = rearrange(f_codes, 'b ( gs gc) -> b gs gc', gs=gs)
    # k is the number of small groups in a group
    w_codes = rearrange(w_codes, '(gs k) gc -> (gs k gc)', k=k)

    # test_w_codes =

    fake_out_codes = []
    # iterate over each sample
    for sample in f_codes:
        repeated_sample = repeat(sample, 'gs gc ->( gs k  gc)', k=k)
        # iterate over each group
        # [gs* k* gc]
        grouped_fake_codes = []
        grouped_fake_codes = fake_comb_keys[repeated_sample, w_codes]
        fake_out_codes.append(grouped_fake_codes)

    # [b, gs* k* gc]
    fake_out_codes = np.stack(fake_out_codes, axis=0)
    # [b, gs* k* gc]
    fake_out_codes = rearrange(fake_out_codes, 'b ( gs k  gc) -> (b gs k) gc', gs=gs, k=k)

    fake_out_keys1, fake_out_keys2 = np.split(fake_out_codes, 2, axis=1)
    out_codes = feature_com_keys[fake_out_keys1, fake_out_keys2]
    out_codes = rearrange(out_codes, '(b gs k) ()-> b (gs k)', gs=gs, k=k)
    return out_codes


def fast_forward_dwpw(f_codes, dw_codes, pw_codes, dw_comb_keys, fake_comb_keys, feature_com_keys):
    dw_out_codes, f_codes = fast_forward_dw(f_codes, dw_codes, dw_comb_keys)
    #some of the fakeout_codes different from f_id3
    pw_out_codes = fast_forward_dw_group(dw_out_codes, pw_codes, fake_comb_keys, feature_com_keys)
    return dw_out_codes, pw_out_codes


# f_codes, out_codes1, dw_codes2, dw_group_codes2, sc_codes, dw_comb_keys2, dw_fake_comb_keys2, fake_sc_comb_keys1, fake_group_com_keys_add, fake_sc_comb_keys_add, feature_com_keys2, has_shortcut
def fast_forward_dwpw_with_sc(f_codes, previous_codes, dw_codes, pw_codes, sc_codes, dw_comb_keys, dw_fake_comb_keys,
                              fake_sc_comb_keys, fake_group_comb_keys_add, fake_sc_comb_keys_add, feature_com_keys,
                              has_shortcut=False):
    batch_size = f_codes.size(0)
    out_codes = fast_forward_dw(previous_codes, dw_codes, dw_comb_keys)
    if has_shortcut:
        fake_dw_group_out_codes = fast_forward_dw_group(out_codes, pw_codes, dw_fake_comb_keys,
                                                        fake_group_comb_keys_add)
        fake_sc_out_codes = fast_forward_dw_group(f_codes, sc_codes, fake_sc_comb_keys, fake_sc_comb_keys_add)
        fake_dw_group_out_codes = rearrange(fake_dw_group_out_codes, ' b n -> (b n)')
        fake_sc_out_codes = rearrange(fake_sc_out_codes, 'b n -> (b n)')
        fake_out_codes = []
        for dw_code, sc_code in zip(fake_dw_group_out_codes, fake_sc_out_codes):
            dw_code = dw_code.item()
            sc_code = sc_code.item()
            out_code = feature_com_keys[(dw_code, sc_code)]
            fake_out_codes.append(torch.LongTensor([out_code]))
        fake_out_codes = torch.stack(fake_out_codes, dim=0)
        fake_out_codes = rearrange(fake_out_codes, '(b n) () -> b n', b=batch_size)
        k = 1
    else:
        fake_out_codes = fast_forward_dw_group(out_codes, pw_codes, dw_fake_comb_keys, feature_com_keys)
    return fake_out_codes


def fast_forward_block(block, f_codes, w_codes_dict, comb_keys_dict):
    # load weights codes
    dw_codes1 = w_codes_dict['dw1']
    dw_group_codes1 = w_codes_dict['dw_group1']

    dw_codes2 = w_codes_dict['dw2']
    dw_group_codes2 = w_codes_dict['dw_group2']

    sc_codes = w_codes_dict['sc']

    # load comb_keys
    dw_comb_keys1 = comb_keys_dict['dw1']
    dw_fake_comb_keys1 = comb_keys_dict['dw1_fake']

    dw_comb_keys2 = comb_keys_dict['dw2']
    dw_fake_comb_keys2 = comb_keys_dict['dw2_fake']

    feature_com_keys1 = comb_keys_dict['feature1']
    feature_com_keys2 = comb_keys_dict['feature2']

    # fake_sc_comb_keys1 = comb_keys_dict['sc1']
    # fake_sc_comb_keys_add = comb_keys_dict['sc2']
    # fake_group_com_keys_add = comb_keys_dict['group_add']

    # [b, out_c, in_c]
    dw_out_codes1, pw_outcodes1 = fast_forward_dwpw(f_codes, dw_codes1, dw_group_codes1, dw_comb_keys1, dw_fake_comb_keys1,
                                   feature_com_keys1)
    has_shortcut = False
    # if block.stride != 1 or block.in_channels != KmBasicBlock.expansion * block.out_channels:
    #     has_shortcut = True
    dw_out_codes2, pw_outcodes2 = fast_forward_dwpw(pw_outcodes1, dw_codes2, dw_group_codes2, dw_comb_keys2, dw_fake_comb_keys2,
                                   feature_com_keys2)
    return pw_outcodes2



def dictionary2numpyarray(dict):
    key_pairs = list(dict.keys())
    last_key_pair = key_pairs[-1]
    n_k1, n_k2 = last_key_pair[0], last_key_pair[1]

    np_dict = np.zeros((n_k1 + 1, n_k2 + 1)).astype('int16') - 1
    for key_pair in key_pairs:
        np_dict[key_pair[0], key_pair[1]] = dict[key_pair]
    # test_np_keys = np.array([[0 , 0], [1, 0], [2 ,0]])
    # test_np_keys1, test_np_keys2 = test_np_keys[:, 0], test_np_keys[:, 1]
    # test_np_values = np_dict[test_np_keys1, test_np_keys2]
    return np_dict

def main(args):
    root = args.root
    results_dir = args.results_dir
    save_path = os.path.join(root, results_dir)
    os.makedirs(save_path, exist_ok=True)
    json_file_name = os.path.join(save_path, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)
    checkpoints_path = os.path.join(save_path, 'checkpoints')
    os.makedirs(checkpoints_path, exist_ok=True)
    sample_output_path = os.path.join(save_path, 'output')
    os.makedirs(sample_output_path, exist_ok=True)

    log_file = os.path.join(save_path, 'log.txt')
    config_logging(log_file)
    logging.info('====>  args{} '.format(args))

    device = "cpu"

    # n_dw_emb, n_pw_emb, n_femb = 2, 3, 4
    # n_dw_emb, n_pw_emb, n_femb = 16, 8, 16
    # n_dw_emb, n_pw_emb, n_femb = 256, 32, 256
    n_dw_emb, n_pw_emb, n_femb = args.n_dw_emb, args.n_pw_emb, args.n_femb
    model = km_resnet9(1, n_dw_emb, n_pw_emb, n_femb, num_classes=10, gs=1).cuda()

    # models = models.load_state_dict(torch.load(args.model_pt_path)['models'])
    initial_input = torch.randn(12, 1, 28, 28).cuda()
    model(initial_input)
    checkpoints = torch.load(args.model_pt_path)['models']
    model.load_state_dict(checkpoints)

    dataset_path = args.dataset_path

    test_ds = MNIST(dataset_path, train=False, transform=transform_MNIST, download=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers)
    for m in model.modules():
        if isinstance(m, Feature_Quantizer):
            first_f_qtz = m.quantizer
            break

    def set_bn_eval(m):
        m.eval()
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    model.apply(set_bn_eval)
    for module in model.modules():
        if isinstance(module, KmConv_parent):
            module.set_use_quant(True)
            module.mode = 'quant'
        if isinstance(module, Feature_Quantizer):
            module.set_use_quant(True)
            module.mode = 'quant'
    loader = tqdm(test_loader)
    ###
    # test original models
    # acc_sum = 0
    # n_sum = 0
    # models = models.cuda()
    # for x, y in loader:
    #     y = y.cuda()
    #     x = x.cuda()
    #     with torch.no_grad():
    #         logits = models(x)
    #         logits = logits.view(logits.size(0), -1)
    #
    #         _, winners = (logits).max(1)
    #         # print(winners[:10])
    #         # print('label', y[:10])
    #         acc = torch.sum((winners == y).int())
    #         # print(winners)
    #         acc_sum += acc.detach().item()
    #         n_sum += y.size(0)
    #         avg_acc = acc_sum / n_sum
    #
    #     loader.set_description(
    #         (
    #             f" real acc:{avg_acc:.5f} ;"
    #
    #         )
    #     )
    # print('real acc', avg_acc)


    # do search once
    model = model.cpu()
    w_codes_dict_list = model.get_w_codes_block_wise()  # validated
    model = model.cuda()
    # # if args.debug_get_comb:
    comb_keys_list = get_comb_for_model(model)
    model = model.cpu()
    torch.save(comb_keys_list, os.path.join(save_path, 'comb_keys_list.pt'))
    torch.save(w_codes_dict_list, os.path.join(save_path, 'w_codes_dict_list.pt'))


    # comb_keys_list = torch.load(os.path.join(save_path, 'comb_keys_list.pt'))
    # w_codes_dict_list = torch.load(os.path.join(save_path, 'w_codes_dict_list.pt'))
    numpy_dict_list = []
    for dict4block in comb_keys_list:
        numpy_dict = {}
        # keys = dict4block.keys()
        for module_key in dict4block:
            if dict4block[module_key] is not None:
                numpy_dict[module_key] = dictionary2numpyarray(dict4block[module_key])
        numpy_dict_list.append(numpy_dict)
    np.savez(os.path.join(save_path, 'comb_keys_list.npz'), numpy_dict_list=numpy_dict_list)
    # numpy_dict_list = np.load(os.path.join(save_path, 'comb_keys_list.npz'))['numpy_dict_list']
    blocks = [block for block in model.modules() if isinstance(block, KmBasicBlock)]


    # fast_forward_block(block, f_codes, w_codes_dict, comb_keys_dict):


    last_f_qtz = model.last_fqtz.quantizer
    avg_pool = model.avg_pool
    last_layer = model.last_layer
    acc_sum = 0
    n_sum = 0

    loader = tqdm(test_loader)
    for x, y in loader:
        y = y.to(device)
        x = x.to(device)

        features_ori = model.conv1(x)
        features = rearrange(features_ori, 'b c h w -> b c (h w)')
        f_codes = first_f_qtz.encode(features).detach().numpy()
        for block, w_codes_dict, numpy_dict in zip(blocks, w_codes_dict_list, numpy_dict_list):
            # models = models.cuda()
            # test_f = rearrange(features_ori[0, torch.LongTensor([14, 22, 15])], 'c h w -> () c (h w)')
            # test_w = models.conv2_x.KmRes0.conv1.dw
            # test_fid = models.fq_dw2.quantizer.encode(test_f)
            # _, f_id1, f_id2, f_id3, f_id4 = block.debug_forward(features_ori.cuda())
            model = model.cpu()
            f_codes = fast_forward_block(block, f_codes, w_codes_dict, numpy_dict)
        f_codes = torch.from_numpy(f_codes).long()
        out_features = last_f_qtz.embed_code(f_codes.to(device))
        h = int(sqrt(out_features.size(2)))
        out_features = rearrange(out_features, 'b c (h w)-> b c h w', h=h)
        output = avg_pool(out_features)
        output = rearrange(output, 'b c () () -> b c ')
        logits = last_layer(output).squeeze()


        logits = logits.view(logits.size(0), -1)
        _, winners = (logits).max(1)
        # print(winners)
        acc = torch.sum((winners == y).int())
        acc_sum += acc.detach().item()
        n_sum += y.size(0)
        avg_acc = acc_sum / n_sum
        # print(avg_acc)
        loader.set_description(
            (
                f" acc:{avg_acc:.5f} ;"
            )
        )
    print('acc', avg_acc)



    k = 1


# comb_keys_list = test_dw_comb(models)
# fake_comb_keys_list, feature_com_keys_list = test_dw_group_comb(models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--dataset_path", default="D:\Dataset\mnist", type=str)
    parser.add_argument("--root", default='', type=str)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--results_dir", default='results', type=str)
    parser.add_argument("--model_pt_path", default='results/checkpoints/classifier_resume.pt', type=str)
    parser.add_argument("--n_dw_emb", default=100, type=int)
    parser.add_argument("--n_pw_emb", default=100, type=int)
    parser.add_argument("--n_femb", default=100, type=int)
    parser.add_argument("--beta", default=0.5, type=float)

    parser.add_argument("--end_class", default=10, type=int)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--wq_epoch", default=0, type=int)
    parser.add_argument("--fq_epoch", default=0, type=int)
    parser.add_argument("--use_amp", default=0, type=int)
    parser.add_argument("--max_norm", default=2, type=int)
    parser.add_argument("--debug_get_comb", default=0, type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)