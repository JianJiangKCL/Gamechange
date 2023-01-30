import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import time
from einops import repeat, rearrange
# from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

class Quantizer(nn.Module):
    #decay=0.99 is default value
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # dxN_e
        embed = torch.randn(dim, n_embed)
        # torch.nn.init.xavier_uniform(embed)
        # self.my_buffer is from self.register_buffer(name, tensor)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

        # using buffer can decrease the gpu cost as the gradients are not traced. but we still need to update .data
        # the summation of chosen vectors for each embedding
        self.register_buffer("acum_embed_sum", torch.zeros_like(self.embed))
        # how many times the embedding is chosen
        self.register_buffer("acum_embed_onehot_sum", torch.zeros(n_embed))
        # tsn = torch.randn((quantize.size()))
        # self.test = nn.Parameter(tsn).to('cuda')

        # self.acum_embed_onehot_sum = torch.zeros(n_embed).cuda()
        # self.acum_embed_sum = torch.zeros_like(self.embed).cuda()

    def zero_buffer(self):

        self.acum_embed_onehot_sum.data = torch.zeros_like(self.acum_embed_onehot_sum)
        self.acum_embed_sum.data = torch.zeros_like(self.embed)


    def update(self):

        embed_sum_norm = self.acum_embed_sum / (self.acum_embed_onehot_sum + self.eps)
        self.embed_avg.mul_(self.decay).add_(embed_sum_norm, alpha=1 - self.decay)
        # self.embed_avg.copy_(embed_sum_norm)
        self.embed.copy_(self.embed_avg)
        # self.cluster_size.data.mul_(self.decay).add_(
        #     self.acum_embed_onehot_sum, alpha=1 - self.decay
        # )
        # # #   an embedding is learnt from its old value and its members' value;
        #
        # self.embed_avg.data.mul_(self.decay).add_(self.acum_embed_sum, alpha=1 - self.decay)
        # n = self.cluster_size.sum()
        # cluster_size = (
        #         (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
        # )
        # embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
        # self.embed.data.copy_(embed_normalized)

    def update_new_emb(self):
        n_emb = self.n_embed
        # [n_emb:] means the new embs
        self.cluster_size.data[n_emb:].mul_(self.decay).add_(
            self.acum_embed_onehot_sum[n_emb:], alpha=1 - self.decay
        )

        self.embed_avg.data[:, n_emb:].mul_(self.decay).add_(self.acum_embed_sum[:, n_emb:], alpha=1 - self.decay)
        #todo this part I'm not sure
        n = self.cluster_size[n_emb:].sum()
        cluster_size = (self.cluster_size[n_emb:] + self.eps) / (n + n_emb * self.eps) * n

        embed_normalized = self.embed_avg[:, n_emb:] / cluster_size.unsqueeze(0)
        self.embed.data[:, n_emb:].copy_(embed_normalized)

    def forward(self, input):
        # for weights, input is out_c, in_c, h*w
        # out_dim, in_dim
        flatten = input.reshape(-1, self.dim)
        # @ is matmul; dist is (x-y)^2 = x^2 - 2xy + y^2
        # B*H*WxN_e
        # @timeit
        def cal_dist():
            dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed  # B*H*WxC matmul* CxN_e => B*H*WxN_e
                + self.embed.pow(2).sum(0, keepdim=True)
            )
            return dist
        # @timeit
        def cal_dist1():
            dist = torch.cdist(flatten, self.embed.permute(1, 0), compute_mode="use_mm_for_euclid_dist")
            return dist

        # dist2 = cal_dist2()
        # dist1 = cal_dist1()
        dist = cal_dist1()
        # values, indices B*H*W
        _, embed_ind = (-dist).max(1)
        #
        # B*H*WxN_e
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)

        # permute input is necessary.
        embed_ind = embed_ind.view(*input.shape[:-1])
        # out_c, in_c  ;
        quantize = self.embed_code(embed_ind)

        if self.training:

            # Note treat B*H*W as positions; C is a set of values in a certain position
            # B*H*WxN_e => N_e; it means how many times an embedding is chosen during this mini-batch
            embed_onehot_sum = embed_onehot.sum(0)
            # Note an embedding's value is learnt from its members' vector values (the avg of members' value)
            #   CxB*H*W matmul* B*H*WxN_e => CxN_e ;
            #
            #  the summation of corresponding dim from selected members for an embedding;
            embed_sum = flatten.transpose(0, 1).contiguous() @ embed_onehot
            ####
            # with torch.no_grad():
            self.acum_embed_onehot_sum.data.add_(embed_onehot_sum)
            self.acum_embed_sum.data.add_(embed_sum)

        # stop gradients to embeddings; inputs are updated by diff to get close to corresponding embedding
        diff = (quantize.detach() - input).pow(2).mean()
        # this sentence bring the gradients caused by quantize and following layers to the input, i.e. former layers.
        # and above codes don't contribute to break-through gradients
        quantize = input + (quantize - input).detach()
        # quantize = quantize.detach()
        return quantize, diff, embed_ind

    def add_zero_emb(self, device):
        zero_embed = torch.zeros([self.dim, 1]).to(device)
        self.embed = torch.cat([self.embed, zero_embed], dim=1)
        self.n_embed = self.n_embed + 1

    def add_emb(self, n_new_emb):
        self.n_new_emb = n_new_emb
        new_embed = torch.randn([self.dim, n_new_emb]).cuda()
        self.embed = torch.cat([self.embed, new_embed], dim=1)
        self.cluster_size = torch.cat([self.cluster_size, torch.zeros(n_new_emb).cuda()], dim=0)
        self.n_embed = self.n_embed + n_new_emb
        self.embed_avg = torch.cat([self.embed_avg, torch.zeros([self.dim, n_new_emb]).cuda()], dim=1)
        self.acum_embed_sum = torch.cat([self.acum_embed_sum, torch.zeros([self.dim, n_new_emb]).cuda()], dim=1)
        self.acum_embed_onehot_sum = torch.cat([self.acum_embed_onehot_sum, torch.zeros(n_new_emb).cuda()], dim=0)

    def add_emb_copied(self, n_new_emb, unique_indices):

        n_unique = unique_indices.size(0)
        unique_emb = self.embed_code(unique_indices.cuda())
        new_embed = torch.zeros([n_new_emb, unique_emb.size(1)]).cuda()
        n_new_emb = n_new_emb
        num_group = n_new_emb // n_unique
        if num_group > 1:
            unique_emb = repeat(unique_emb, ' n d -> (repeat n) d', repeat=num_group)
        n_unique = unique_emb.size(0)
        n_left = n_new_emb - n_unique

        if n_left > 0:
            n_left = n_new_emb - n_unique
            new_embed[:len(unique_emb), :] = unique_emb
            new_embed[len(unique_emb):, :] = unique_emb[:n_left, :]
        else:
            new_embed = unique_emb[:self.n_new_emb, :]
        new_embed = rearrange(new_embed, 'n_emb dim -> dim n_emb')

        self.n_new_emb = n_new_emb
        # [dim, n_emb]
        self.embed = torch.cat([self.embed, new_emb], dim=1)
        self.cluster_size = torch.cat([self.cluster_size, torch.zeros(n_new_emb).cuda()], dim=0)
        self.n_embed = self.n_embed + n_new_emb
        self.embed_avg = torch.cat([self.embed_avg, torch.zeros([self.dim, n_new_emb]).cuda()], dim=1)
        self.acum_embed_sum = torch.cat([self.acum_embed_sum, torch.zeros([self.dim, n_new_emb]).cuda()], dim=1)
        self.acum_embed_onehot_sum = torch.cat([self.acum_embed_onehot_sum, torch.zeros(n_new_emb).cuda()], dim=0)

    def reset_nemb(self, nemb):
        self.n_embed = nemb
        self.embed = torch.randn(self.dim, nemb)
        self.embed_avg = self.embed.clone()
        self.cluster_size = torch.zeros(nemb).cuda()
        self.acum_embed_sum = torch.zeros([self.dim, nemb]).cuda()
        self.acum_embed_onehot_sum = torch.zeros(nemb).cuda()

    def keep_emb_and_delete_others(self, emb_ind_to_keep):
        self.n_embed = len(emb_ind_to_keep)
        self.embed = self.embed[:, emb_ind_to_keep]
        self.cluster_size = self.cluster_size[emb_ind_to_keep]
        self.embed_avg = self.embed_avg[:, emb_ind_to_keep]
        self.acum_embed_sum = self.acum_embed_sum[:, emb_ind_to_keep]
        self.acum_embed_onehot_sum = self.acum_embed_onehot_sum[emb_ind_to_keep]


    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1).contiguous())

    def embed_sparse_code(self, embed_id, sparse_codes):
        # [ n, dim]
        embed = self.embed.transpose(0, 1).contiguous()
        zeros = torch.zeros(sparse_codes.size(0), embed.size(1)).cuda()
        embed[sparse_codes, :] = zeros
        return F.embedding(embed_id, embed)

    def embed_sparse_code_high(self, embed_id, sparse_codes):
        # [ n, dim]
        embed = self.embed.transpose(0, 1).contiguous()
        zeros = torch.zeros(sparse_codes.size(0), embed.size(1)).cuda()
        embed[sparse_codes, :] = zeros
        return F.embedding(embed_id, embed)

    def encode(self, input):
        # out_dim, in_dim
        flatten = input.reshape(-1, self.dim)
        # @ is matmul; dist is (x-y)^2 = x^2 - 2xy + y^2
        # B*H*WxN_e
        # @timeit
        def cal_dist1():
            dist = torch.cdist(flatten, self.embed.permute(1, 0), compute_mode="use_mm_for_euclid_dist")
            return dist

        dist = cal_dist1()
        # values, indices B*H*W
        _, embed_ind = (-dist).max(1)

        embed_ind = embed_ind.view(*input.shape[:-1])
        return embed_ind


    def embed_code_straight_through(self, input, embed_id):
        quantize = F.embedding(embed_id, self.embed.transpose(0, 1).contiguous())
        quantize = input + (quantize - input).detach()
        return quantize

def weight_quantization(W, Q, in_mem, out_mem, bias=False):

    # dim, n_embed = Q.dim, Q.n_embed
    in_features, out_features = W.in_features, W.out_features

    in_g = int(in_features / in_mem)
    out_g = int(out_features / out_mem)
    # tmp_fc = torch.nn.Linear(in_features, out_features).to('cuda')
    tmp_fc = copy.deepcopy(W).to('cuda')
    groups = []
    tmp_weight = W.weight.t()
    for g in range(0, out_g):
        indices = torch.arange(g * out_mem, (g + 1) * out_mem, 1).to('cuda')

        w_outmem = tmp_weight.index_select(1, indices)
        w_group = w_outmem.chunk(int(in_g), dim=0)
        # groups = tor.stack(w_group)
        groups.append(torch.stack(w_group))
        k = 1
    groups = torch.stack(groups)
    groups_ = groups.view(*groups.size()[0:-2], -1)

    # weight shape [out,in] -> [in, out] [ingroups * outgroups, d_emb]
    #todo delete .type(torch.int64)
    # tmp_weight = W.weight.type(torch.int64)
    # print(tmp_weight.stride())

    # tmp_weight = W.weight.view(int(512/outmem), int(512/inmem))
    # [ in_group* out_group , d_emb]
    quant_weight, diff, ids = Q(groups_)
    # quant_weight = quant_weight.view(out_features, in_features )

    groups_t = quant_weight.view(groups.size(0), groups.size(1) * groups.size(2), -1)

    back_weights = []
    for i in range(groups_t.size(1)):
        in_w = groups_t[:, i, :].flatten()
        back_weights.append(in_w)
    back_weights = torch.stack(back_weights, dim=1)


    tmp_fc.weight.data = back_weights
    # this is used to bring the gradients of tmp_fc.bias to W.bias to update bias.
    if bias:
        tmp_bias = W.bias #+ (tmp_fc.bias - W.bias).detach()
        tmp_fc.bias = tmp_bias
    # w/o tmp_fc, i.e., replacing the W.weight directly will cause nan problem
    return diff, tmp_fc, ids

class WQ_MLP(nn.Module):
    def __init__(self, n_embed, n_inmember=2**4, bias=True):
        super(WQ_MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        self.n_inmember = [7, 16 , 16]
        self.n_outmember = [4, 1, 1]
        # self.n_inmember = [16, 16, 16]
        # self.n_outmember = [2, 2, 2]
        input_size = 28*28
        # linear layer (784 -> hidden_1)
        # todo a layer creater
        self.fc1 = nn.Linear(input_size, hidden_1, bias=bias)
        self.quant1 = Quantize(int(input_size/self.n_inmember[0])*self.n_outmember[0], n_embed)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2, bias=bias)
        self.quant2 = Quantize(int(hidden_1/self.n_inmember[1])*self.n_outmember[1], n_embed)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10, bias=bias)
        self.quant3 = Quantize(int(hidden_2/self.n_inmember[2])*self.n_outmember[2], n_embed)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        # self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        # diff1 = weight_quantization(self.fc1, self.quant1)
        # n_outgroup = self.n_outmember
        # for shared 49
        diff1, q_fc1, _ = weight_quantization(self.fc1, self.quant1, self.n_inmember[0], self.n_outmember[0])
        x = F.relu(q_fc1(x))
        # x = F.relu(self.fc1(x))
        # add dropout layer
        # x = self.dropout(x)
        # add hidden layer, with relu activation function
        diff2, q_fc2, _ = weight_quantization(self.fc2, self.quant2, self.n_inmember[1], self.n_outmember[1])
        # x = F.relu(self.fc2(x))
        x = F.relu(q_fc2(self.bn1(x)))#x))#
        # add dropout layer
        # x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        # x_tmp = self.fc3(x)
        # diff3, q_fc3, _ = weight_quantization(self.fc3, self.quant3,  self.n_inmember[2], self.n_outmember[2])
        # x = q_fc3(self.bn2(x))#x)#
        # x = x_tmp + (x - x_tmp).detach() #nan problem #w/o cannot

        # x = torch.mean(torch.stack([x , x_tmp], dim=1), dim=1)
        return x, diff1 + diff2 #+ diff3 #+  # +diff1 + diff2


class AQ_MLP(nn.Module):
    def __init__(self, n_embed, n_inmember=2**4, bias=False):
        super(AQ_MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # todo set n_inmem and out_mem as args
        self.n_inmember = [7, 8, 16]
        self.n_outmember = [4, 2, 1]
        self.n_ingroup = [int(hidden_1/ i) for i in self.n_inmember]
        self.n_outgroup = [int(hidden_2/ i ) for i in self.n_outmember]
        # self.n_inmember = [16, 16, 16]
        # self.n_outmember = [2, 2, 2]
        input_size = 28*28
        # linear layer (784 -> hidden_1)
        # todo a layer creater
        self.fc1 = nn.Linear(input_size, hidden_1, bias=bias)
        self.quant1 = Quantize(int(self.n_inmember[0]*self.n_outmember[0]), n_embed)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2, bias=bias)
        self.quant2 = Quantize(int(self.n_inmember[1]*self.n_outmember[1]), n_embed)

        self.fq21 = Quantize(self.n_inmember[1], n_embed)
        self.fq22 = Quantize(self.n_outmember[1], n_embed)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10, bias=bias)
        self.quant3 = Quantize(self.n_inmember[2]*self.n_outmember[2], n_embed)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        # self.dropout = nn.Dropout(0.2)

    def cal_weights_ids(self):
        _,_, ids = weight_quantization(self.fc2, self.quant2, self.n_inmember[1], self.n_outmember[1])
        # ids = ids.view(self.n_outgroup[1], self.n_ingroup[1]).permute(1, 0)
        self.fc2_ids = ids
        return ids

    def get_quants_ids(self, x):
        B = x.size(0)
        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))
        # [B, n_in, C/n_in]
        _, _, ids = self.fq21(x.view(B, self.n_inmember[1], -1).permute(0, 2, 1))

        return ids

    # def get_pred_using_ids(self, ids):
    #     quant = self.fq22.embed_code(ids)
    #     B = quant.size(0)
    #     quant = quant.permute(0, 2, 1).view(B, -1)
    #     x = self.fc3(quant)

    def pred_using_features(self, x):
        B = x.size(0)
        quant_x2, diff_f22, _ = self.fq22(x.view(B, self.n_outmember[1], -1).permute(0, 2, 1))
        # add dropout layer
        # x = self.dropout(x)
        # add output layer
        quant_x2 = quant_x2.permute(0, 2, 1).view(B, -1)
        # x = self.fc3(quant_x2)

        # x_tmp = self.fc3(x)
        diff3, q_fc3, _ = weight_quantization(self.fc3, self.quant3, self.n_inmember[2], self.n_outmember[2])
        x = q_fc3(self.bn2(quant_x2))  # x)#
        return x

    def forward(self, x):
        # flatten image input
        B = x.size(0)
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        # diff1 = weight_quantization(self.fc1, self.quant1)
        # n_outgroup = self.n_outmember
        # for shared 49
        # diff1, q_fc1, _ = weight_quantization(self.fc1, self.quant1, self.n_inmember[0], self.n_outmember[0])
        # x = F.relu(q_fc1(x))
        x = F.relu(self.fc1(x))
        # add dropout layer
        # x = self.dropout(x)
        # add hidden layer, with relu activation function

        # [B, n_in, C/n_in]
        quant_x, diff_f21, _ = self.fq21(x.view(B, self.n_inmember[1], -1).permute(0,2,1))
        s_time = time.time()
        diff2, q_fc2, _ = weight_quantization(self.fc2, self.quant2, self.n_inmember[1], self.n_outmember[1])

        quant_x = quant_x.permute(0,2,1).view(B,-1)
        # x = F.relu(self.fc2(x))
        # [B,512]
        x = F.relu(q_fc2(self.bn1(quant_x)))#x))#
        e_time = time.time()
        print(e_time - s_time)
        quant_x2, diff_f22, _ = self.fq22(x.view(B, self.n_outmember[1], -1).permute(0,2,1))
        # add dropout layer
        # x = self.dropout(x)
        # add output layer
        quant_x2 = quant_x2.permute(0,2,1).view(B, -1)
        x = self.fc3(quant_x2)

        # x_tmp = self.fc3(x)
        # diff3, q_fc3, _ = weight_quantization(self.fc3, self.quant3,  self.n_inmember[2], self.n_outmember[2])
        # x = q_fc3(self.bn2(quant_x2))#x)#
        # x = x_tmp + (x - x_tmp).detach() #nan problem #w/o cannot

        # x = torch.mean(torch.stack([x , x_tmp], dim=1), dim=1)
        return x, diff_f21 + diff_f22 + diff2 #+ diff3 #+  # +diff1 + diff2


def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('Function "{name}" took {time} seconds to complete.'.format(name=f.__name__, time=te-ts))
        # print( 'func:%r args:[%r, %r] took: %2.4f sec' % \
        #   (f.__name__, args, kw, te-ts))
        return result

    return timed

class GroupedWeight_MLP(nn.Module):
    def __init__(self, in_g, out_g, in_mem, out_mem, n_embed):
        super(GroupedWeight_MLP, self).__init__()
        dim_quant = in_mem * out_mem
        self.in_g = in_g
        self.out_g = out_g
        self.in_mem = in_mem
        self.out_mem = out_mem
        self.n_embed = n_embed
        self.weight = nn.Parameter(torch.Tensor(in_g, out_g, in_mem, out_mem))
        torch.nn.init.xavier_uniform(self.weight)
        self.quantizer = Quantize(dim_quant, n_embed)
        # self.get_w_ids()

    # doesn't work; models.eval() didn't call this
    def eval(self):
        self.get_w_ids()
        self.train(False)

    #
    def forward(self, x, use_qw=False):
        B = x.size(0)
        x = x.view(B, self.in_g, self.in_mem)
        # x is [B,in_g, in_mem] -> [ in_g, B, in_mem]
        x = x.permute(1, 0, 2).contiguous()
        if use_qw:
        # q_w quant_weight is [in_g, out_g, d_q]
            q_w, diff, _ = self.quantizer(self.weight.view(self.in_g, self.out_g, -1))
            # [in_g, out_g, d_q] -> [in_g, out_g, in_mem, out_mem] -> [in_g, in_mem, out_g, out_mem] - > [in_g, in_mem, out_d]
            q_w = q_w.view(*self.weight.size()).permute(0, 2, 1, 3).reshape(self.in_g, self.in_mem, -1)
        else:
            q_w = self.weight.permute(0, 2, 1, 3).reshape(self.in_g, self.in_mem, -1)
            diff = None
        # @timeit
        def grouped_mm(x, q_w):
            res = []
            for v, w in zip(x, q_w):
                # v is [B, in_mem] , w is [ in_mem, out_d]; res is [B,out_d]
                tmp_res = v @ w
                res.append(tmp_res)
            # [B, in_g, out_d]
            res = torch.stack(res, dim=1)
            return res

        # @timeit
        #     def grouped_mm2(x, q_w):
        #         # results are not equal to group_mm, speed are similar
        #         # x is [B,in_g, in_mem] -> [B,in_g, in_mem, out_d]
        #         x = x.permute(1, 0,2).unsqueeze(-1).repeat(1,1,1, 512)
        #         #  [in_g, in_mem, out_d] -> [B,in_g, in_mem, out_d]
        #         q_w = q_w.unsqueeze(0).repeat(B,1,1,1 )
        #         # B, in_g, out_d
        #         res = x.mul(q_w).sum(dim=1)
        #         return res
        res = grouped_mm(x, q_w)
        res = res.sum(dim=1)
        return res, diff

    def efficient_forward(self, f_ids, quant_f):
        mini_B = 128
        # f_ids is [B, in_g]
        # q_w quant_weight is [in_g, out_g, d_q];  w_ids is [in_g, out_g]
        q_w, diff, w_ids = self.quantizer(self.weight.view(self.in_g, self.out_g, -1))

        # combinations
        # @timeit 0-100; 0 -100 * 100 +1;
        def get_comb(f_ids, w_ids):
            # because generally the shape of w_ids [in_g, out_g] is smaller than f_ids,
            # manipulating w_ids is computationally cheaper
            w_ids = (w_ids * self.n_embed + 1)
            # [B, in_g] - > [B, in_g, out_g]
            f_ids2 = f_ids.unsqueeze(-1).repeat(1, 1, self.out_g)
            w_ids2 = w_ids.expand_as(f_ids2)
            # vi = vi * self.n_embed + 1
            vw = f_ids2 + w_ids2
            return vw

        # [B, in_g, out_g]
        vw = get_comb(f_ids, w_ids)
        # [n_valid]
        unique_vw = vw.unique()

        valid_v = ((unique_vw - 1) % self.n_embed).type(torch.int64)
        valid_w = ((unique_vw - 1 - valid_v) / self.n_embed).type(torch.int64)

        # [n_val, d_q] or [n_val, in_mem, out_mem]
        weights = self.quantizer.embed_code(valid_w)  # .to('cuda'))
        # [n_val, in_mem]
        features = quant_f.embed_code(valid_v)  # .to('cuda'))

        # @timeit
        def get_val_comp(features, weights):
            # the results are eqault to get_val_comp, but way more faster
            # [n_val, in_mem, out_mem]
            w = weights.view(-1, self.in_mem, self.out_mem)
            # [n_val, in_mem, out_mem]
            f = features.unsqueeze(-1).repeat(1, 1, self.out_mem)
            # [n_val, out_mem]
            valid_comp = f.mul(w).sum(dim=1)
            return valid_comp

        valid_comp = get_val_comp(features, weights)
        from_values = unique_vw
        to_values = torch.arange(unique_vw.size(0)).to('cuda')

        # @timeit
        def get_mapped_vw(vw):
            # 0.0239365
            vw = vw.flatten()
            idces = torch.bucketize(vw, from_values)
            out = to_values[idces]
            return out

        pos = get_mapped_vw(vw)
        # res = valid_comp[pos].view(B, self.in_g, self.out_g, self.out_mem).view(B, self.in_g, self.out_g * self.out_mem)
        res = valid_comp[pos].view(f_ids.size(0), self.in_g, self.out_g * self.out_mem)
        res = res.sum(dim=1)
        return res, diff
    def get_w_ids(self):
        _, _, w_ids = self.quantizer(self.weight.view(self.in_g, self.out_g, -1))
        self.w_ids = w_ids.to('cuda')
        return w_ids

    def efficient_inference(self, f_ids, quant_f):
        # w is in_g, out_g, in_mem, out_mem)
        # w_ids = self.w_ids
        # w_ids [in_g, out_g]
        _, _, w_ids = self.quantizer(self.weight.view(self.in_g, self.out_g, -1))
        self.w_ids = w_ids.to('cuda')
        # combinations
        # @timeit
        def get_comb(f_ids, w_ids):
            # todo it depends
            # because generally the shape of w_ids [in_g, out_g] is smaller than f_ids [B, in_g],
            # manipulating w_ids is computationally cheaper
            w_ids = w_ids * self.n_embed + 1
            # [B, in_g] - > [B, in_g, out_g]
            f_ids2 = f_ids.unsqueeze(-1).repeat(1, 1, self.out_g)
            # use expand_as to calibrate the dimension (repeat should also work)
            w_ids2 = w_ids.expand_as(f_ids2)
            # vi = vi * self.n_embed + 1
            vw = f_ids2 + w_ids2
            return vw

        # [B, in_g, out_g]
        vw = get_comb(f_ids, w_ids)
        # [n_valid]
        unique_vw = vw.unique()

        valid_v = ((unique_vw - 1) % self.n_embed).type(torch.int64)
        valid_w = ((unique_vw - 1 - valid_v) / self.n_embed).type(torch.int64)

        # [n_val, d_q] or [n_val, in_mem, out_mem]
        weights = self.quantizer.embed_code(valid_w)  # .to('cuda'))
        # [n_val, in_mem]
        features = quant_f.embed_code(valid_v)  # .to('cuda'))

        # @timeit
        def get_val_comp(features, weights):
            # the results are eqault to get_val_comp, but way more faster
            # [n_val, in_mem, out_mem]
            w = weights.view(-1, self.in_mem, self.out_mem)
            # [n_val, in_mem, out_mem]
            f = features.unsqueeze(-1).repeat(1, 1, self.out_mem)
            # [n_val, out_mem]
            valid_comp = f.mul(w).sum(dim=1)
            return valid_comp

        valid_comp = get_val_comp(features, weights)
        from_values = unique_vw
        to_values = torch.arange(unique_vw.size(0)).to('cuda')
        # @timeit
        def get_mapped_vw(vw):
            # 0.0239365
            vw = vw.flatten()
            idces = torch.bucketize(vw, from_values)
            out = to_values[idces]
            return out

        pos = get_mapped_vw(vw)
        # res = valid_comp[pos].view(B, self.in_g, self.out_g, self.out_mem).view(B, self.in_g, self.out_g * self.out_mem)
        res = valid_comp[pos].view(f_ids.size(0), self.in_g, self.out_g * self.out_mem)
        res = res.sum(dim=1)
        return res

class TEST_MLP(nn.Module):
    def __init__(self, n_embed, end_class=10, n_inmember=2**4, bias=False):
        super(TEST_MLP, self).__init__()
        # number of hidden nodes in each layer (512)
        input_size = 28 * 28

        hidden_1 = 512
        hidden_2 = 512
        output_size = end_class
        inputs_d = [input_size, hidden_1, hidden_2, output_size]
        # todo set n_inmem and out_mem as args
        self.n_inmember = [28, 8, 2]
        self.n_outmember = [8, 2, 1]
        self.n_ingroup = [int(d / n) for d, n in zip(inputs_d[:-1], self.n_inmember)]
        self.n_outgroup = [int(d / n) for d, n in zip(inputs_d[1:], self.n_outmember)]
        # self.n_inmember = [16, 16, 16]
        # self.n_outmember = [2, 2, 2]

        # linear layer (784 -> hidden_1)
        # todo a layer creater
        self.fc1 = nn.Linear(input_size, hidden_1, bias=bias)
        # self.quant1 = Quantize(int(self.n_inmember[0]*self.n_outmember[0]), n_embed)
        self.gw_fc1 = GroupedWeight_MLP(self.n_ingroup[0], self.n_outgroup[0], self.n_inmember[0], self.n_outmember[0], n_embed)
        self.gw_fc2 = GroupedWeight_MLP(self.n_ingroup[1], self.n_outgroup[1], self.n_inmember[1], self.n_outmember[1], n_embed)

        self.gw_fc3 = GroupedWeight_MLP(self.n_ingroup[2], self.n_outgroup[2], self.n_inmember[2], self.n_outmember[2],
                                        n_embed)
        self.fq21 = Quantize(self.n_inmember[1], n_embed)
        self.fq22 = Quantize(self.n_outmember[1], n_embed)
        self.p = 0.5
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10, bias=bias)
        self.quant3 = Quantize(self.n_inmember[2]*self.n_outmember[2], n_embed)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        # self.dropout = nn.Dropout(0.2)


    def forward(self, x, use_wq=True):
        # flatten image input
        B = x.size(0)
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        # diff1 = weight_quantization(self.fc1, self.quant1)
        # n_outgroup = self.n_outmember
        # for shared 49
        # diff1, q_fc1, _ = weight_quantization(self.fc1, self.quant1, self.n_inmember[0], self.n_outmember[0])
        # x = F.relu(q_fc1(x))

        # x = F.relu(self.fc1(x))
        # x = x.view(B, self.gw_fc1)
        # use_wq = torch.rand(1) < self.p

        use_wq = True
        x, diff1 = self.gw_fc1(x, use_qw=use_wq)
        x = F.relu(x)
        # add dropout layer

        # x = self.dropout(x)
        # add hidden layer, with relu activation function

        # [B, n_in, C/n_in]

        # quant_x, diff_f21, f_ids = self.fq21(x.view(B, self.n_inmember[1], -1).permute(0, 2, 1))

        #  [B, in_g, in_mem]

        # x = F.relu(self.fc2(x))
        # [B,512]
        # x = F.relu(q_fc2(self.bn1(quant_x)))#x))#
        x, diff2 = self.gw_fc2(x, use_qw =use_wq)
        # if self.training:
        #     # x, diff2 = self.gw_fc2.efficient_forward(f_ids, self.fq21)
        #     x, diff2 = self.gw_fc2(quant_x)
        # else:
        #     # x, diff2 = self.gw_fc2(quant_x)
        #     x = self.gw_fc2.efficient_inference(f_ids, self.fq21)


        x = F.relu(x)
        # quant_x2, diff_f22, _ = self.fq22(x.view(B, self.n_outmember[1], -1).permute(0,2,1))
        # # add dropout layer
        # # x = self.dropout(x)
        # # add output layer
        # quant_x2 = quant_x2.permute(0,2,1).view(B, -1)
        # x = self.fc3(quant_x2)
        # x = self.fc3(x)
        x, diff3 = self.gw_fc3(x, use_qw=use_wq)
        if use_wq:
            return x,  diff2 + diff1 + diff3#+ diff_f22 diff_f21 +
        else:
            return x, 0
