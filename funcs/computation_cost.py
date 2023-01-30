import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.profiler import profile, record_function, ProfilerActivity
import sys
sys.path.append('..')
from models.AqResnet import QuantHelper, QUANT_TYPE
from thop import profile as thop_profile
from thop import clever_format

class Quantizer(nn.Module):
	# decay=0.99 is default value
	def __init__(self, dim, n_embed, decay=0.99, target=QUANT_TYPE.WEIGHT, eps=1e-5):
		super().__init__()
		self.dim = dim
		self.n_embed = n_embed
		self.decay = decay
		self.target = target
		self.eps = eps
		# self.diff = 0.

		# dxN_e
		embed = torch.randn(dim, n_embed)
		# torch.nn.init.xavier_uniform(embed)
		self.register_buffer("embed", embed)
		self.register_buffer("cluster_size", torch.zeros(n_embed))
		self.register_buffer("embed_avg", embed.clone())
		self.register_buffer("acum_embed_sum", torch.zeros_like(self.embed))
		self.register_buffer("acum_embed_onehot_sum", torch.zeros(n_embed))

	def set_decay(self, decay):
		self.decay = decay

	def forward(self, input):
		flatten = input.reshape(-1, self.dim)
		dist = torch.cdist(flatten, self.embed.permute(1, 0), compute_mode="use_mm_for_euclid_dist")
		_, embed_ind = dist.min(1)

		# B*H*WxN_e
		embed_ind = embed_ind.view(*input.shape[:-1])
		quantize = self.embed_code(embed_ind)

		return quantize

	def embed_code(self, embed_id):
		return F.embedding(embed_id, self.embed.transpose(0, 1).contiguous())

class StandardConv2d(torch.nn.Module):
	def __init__(self, in_channels, d_j, kernel_size, stride, padding, bias=False):
		super().__init__()
		self.conv = torch.nn.Conv2d(in_channels, d_j, kernel_size, stride, padding, bias=bias)
	def forward(self, x):
		return self.conv(x)


class DepthwiseSeparableConv2d(torch.nn.Module):
	def __init__(self, in_channels, d_j, kernel_size, stride, padding, bias=False):
		super().__init__()
		self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
		self.pointwise = torch.nn.Conv2d(in_channels, d_j, 1, 1, 0, bias=bias)

	def forward(self, x):
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x


class FeatureQuantizer(QuantHelper):
	def __init__(self, dim, n_emb, out_channel, decay=0.99):
		super().__init__()
		self.use_quant = True
		self.target = QUANT_TYPE.FEATURE
		self.quantizer = Quantizer(dim, n_emb, decay)
		self.out_channel = out_channel
		self.quantizer.eval()

	def normal_forward(self, x):
		return x

	def quant_forward(self, x):
		B, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b c (h w)')
		quantized_x = self.quantizer(x)
		quantized_x = rearrange(quantized_x, 'b c (h w) -> b c h w', h=H, w=W)


		# quantized_x = sum(quantized_x)
		return quantized_x


def cal_flop(model, inputs):
	flops = 0

	with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
		with record_function("model_inference"):
			model(inputs)
			# print flops
	print(prof.key_averages().table(sort_by="cpu_time_total"))


def cal_flop_thop(model, inputs):
	macs, params = thop_profile(model, inputs=(inputs, ))
	flops, params = clever_format([2*macs, params], "%.3f")
	print(flops, params)


def theoretical_standard(k, d_i, d_j, h, w):
	# k: kernel size
	# d_i: input channel
	# d_j: output channel
	# h: height
	# w: width
	print(f'{2 * k * k * d_i * d_j * h * w / 1000/1000} MFlops')


def theoretical_depthwiseSeparate(k, d_i, d_j, h, w):
	# k: kernel size
	# d_i: input channel
	# d_j: output channel
	# h: height
	# w: width
	print(f'{2*(k * k + d_j) * d_i * h * w /1000/1000} MFlops')


#todo not exactly right compared to profiler results
def theoretical_quantization(n_emb, d_i, h, w):
	d_emb = h*w

	print(f'{(2 * d_i )* n_emb  * d_emb / 1000 } KFlops')



d_i = 64
h = 32
d_j = 64
n_emb = 64
inputs = torch.randn(1, d_i, h, h)
standard = StandardConv2d(d_i, d_j, 3, 1, 1)
depthwise = DepthwiseSeparableConv2d(d_i, d_j, 3, 1, 1)

quant = FeatureQuantizer(h*h, n_emb, d_j)

# cal_flop(standard, inputs)
# theoretical_standard(3, d_i, d_j, h, h)
#
# cal_flop(depthwise, inputs)
# theoretical_depthwiseSeparate(3, d_i, d_j, h, h)
#
# cal_flop_thop(standard, inputs)
# cal_flop_thop(depthwise, inputs)
cal_flop(quant, inputs)
theoretical_quantization(n_emb, d_i, h, h)


# def make_tensors():
#     return [torch.randn(5, 5) for _ in range(1000)]
#
#
# def profile_action(label, action):
#     print(label)
#     list_of_tensors = make_tensors()
#     with torch.autograd.profiler.profile(
#         profile_memory=True, record_shapes=True, with_flops=True
#     ) as prof:
#         action(list_of_tensors)
#
#     print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
#
#
# profile_action("Case A:", lambda tensors: torch.sum(torch.stack(tensors), dim=0))
# profile_action("Case B:", lambda tensors: sum(tensors))


# d_i = 3
# 3.539 M
# 449 K
# 3154 K

# d_i = 64
#  75.497  M
# 9.568 M
# 65.7 M