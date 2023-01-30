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
from funcs.utils_funcs import set_seed, load_state_dict_flexible_
from einops import rearrange, repeat
from numpy import sqrt
from models.funcs import create_backbone
import wandb
from models.AqMethod import AqMethod
from funcs.setup import parse_args, set_logger, set_trainer
from data import create_dataset


def direct_comb(in_qtz, w_qtz, out_qtz, layer):
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
    out = F.conv2d(repeated_in_embs, repeated_w_embs, None, layer.conv.stride, layer.conv.padding, groups=n_in * n_w)
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

# search for Quant_dwpw2
def get_BasicBlock_comb(dwpw):
    quantizer_dw = dwpw.quantizer_dw
    quantizer_pw = dwpw.quantizer_pw
    quantizer_dw2 = dwpw.quantizer_dw2
    quantizer_pw2 = dwpw.quantizer_pw2
    fq_dw = dwpw.feat_quantizer_dw.quantizer
    fq_pw = dwpw.feat_quantizer_pw.quantizer
    fq_dw2 = dwpw.feat_quantizer_dw2.quantizer
    fq_pw2 = dwpw.feat_quantizer_pw2.quantizer
    # get layer to pass its stride and padding
    dw_conv = dwpw.dw_conv
    pw_conv = dwpw.pw_conv
    dw_conv2 = dwpw.dw_conv2
    dw_comb_keys = direct_comb(fq_dw, quantizer_dw, fq_pw, dw_conv)

    pw_comb_keys = direct_comb(fq_pw, quantizer_pw, fq_dw2, pw_conv)

    dw2_comb_keys = direct_comb(fq_dw2, quantizer_dw2, fq_pw2, dw_conv2)
    return dw_comb_keys, pw_comb_keys, dw2_comb_keys


def get_comb_Block_2conv(block):
    quantizer_pw = block.quantizer_pw
    quantizer_pw2 = block.quantizer_pw2

    pw_conv = block.pw_conv
    pw_conv2 = block.pw_conv2

    fq_pw = block.feat_quantizer_pw.quantizer
    fq_pw2 = block.feat_quantizer_pw2.quantizer

    pw_comb_keys = direct_comb(fq_pw, quantizer_pw, fq_pw2, pw_conv)

    return pw_comb_keys


def get_comb_for_model(model):
    blocks = []
    for name, m in model.named_modules():
        if isinstance(m, Quant_dwpw2):
            blocks.append(m)
    next_block_first_f_qtz = None
    comp_keys_list = []
    for i, block in enumerate(blocks):

        print(f'{i} block keys building finished')

        dw_comb_keys, pw_comb_keys, dw2_comb_keys = get_BasicBlock_comb(block)

        comb_dict = {'dw1': dw_comb_keys,
                     'pw_comb_keys': pw_comb_keys,
                     'dw2': dw2_comb_keys}
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
    save_path = args.results_dir + '/fast_forward'
    os.makedirs(save_path, exist_ok=True)

    logging.info('====>  args{} '.format(args))

    test_dataset, _, _ = create_dataset(args.dataset, args.dataset_path, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers,
                             pin_memory=True, shuffle=False)
    device = "cpu"

    model = create_backbone(args)
    model = model.cuda()
    model.eval()
    # initial models to create the feature quantizer
    initial_input = torch.randn(12, 1, 28, 28).cuda()
    model(initial_input)
    checkpoints = torch.load(args.finetune)
    k=1

    load_state_dict_flexible_(model, checkpoints['state_dict'])

    def set_bn_eval(m):
        m.eval()
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    model.apply(set_bn_eval)

    loader = tqdm(test_loader)

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
    # blocks = [block for block in models.modules() if isinstance(block, KmBasicBlock)]



    # avg_pool = models.avg_pool
    # last_layer = models.last_layer
    # acc_sum = 0
    # n_sum = 0
    #
    # loader = tqdm(test_loader)
    # for x, y in loader:
    #     y = y.to(device)
    #     x = x.to(device)
    #
    #     features_ori = models.conv1(x)
    #     features = rearrange(features_ori, 'b c h w -> b c (h w)')
    #     f_codes = first_f_qtz.encode(features).detach().numpy()
    #     for block, w_codes_dict, numpy_dict in zip(blocks, w_codes_dict_list, numpy_dict_list):
    #         # models = models.cuda()
    #         # test_f = rearrange(features_ori[0, torch.LongTensor([14, 22, 15])], 'c h w -> () c (h w)')
    #         # test_w = models.conv2_x.KmRes0.conv1.dw
    #         # test_fid = models.fq_dw2.quantizer.encode(test_f)
    #         # _, f_id1, f_id2, f_id3, f_id4 = block.debug_forward(features_ori.cuda())
    #         models = models.cpu()
    #         f_codes = fast_forward_block(block, f_codes, w_codes_dict, numpy_dict)
    #     f_codes = torch.from_numpy(f_codes).long()
    #     out_features = last_f_qtz.embed_code(f_codes.to(device))
    #     h = int(sqrt(out_features.size(2)))
    #     out_features = rearrange(out_features, 'b c (h w)-> b c h w', h=h)
    #     output = avg_pool(out_features)
    #     output = rearrange(output, 'b c () () -> b c ')
    #     logits = last_layer(output).squeeze()
    #
    #
    #     logits = logits.view(logits.size(0), -1)
    #     _, winners = (logits).max(1)
    #     # print(winners)
    #     acc = torch.sum((winners == y).int())
    #     acc_sum += acc.detach().item()
    #     n_sum += y.size(0)
    #     avg_acc = acc_sum / n_sum
    #     # print(avg_acc)
    #     loader.set_description(
    #         (
    #             f" acc:{avg_acc:.5f} ;"
    #         )
    #     )
    # print('acc', avg_acc)



    k = 1


# comb_keys_list = test_dw_comb(models)
# fake_comb_keys_list, feature_com_keys_list = test_dw_group_comb(models)

if __name__ == "__main__":
    args = parse_args()
    # set random seed
    set_seed(args.seed)
    # pl.seed_everything(args.seed)
    print(args)

    main(args)