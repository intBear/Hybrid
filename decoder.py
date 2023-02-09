import torch
from entropy_models import *
import torch.nn as nn
import math


class SynthesisTransform(nn.Module):
	def __init__(self, L, dim_hidden, num_layers):
		super().__init__()
		layers = [nn.Linear(L, dim_hidden), nn.ReLU()]
		if num_layers > 2:
			for i in range(num_layers - 2):
				layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
		layers += [nn.Linear(dim_hidden, 3)]
		self.net = nn.Sequential(*layers)

	def forward(self, z):
		return self.net(z)


#  暂时不知道怎么用MLP实现context prediction，用masked_CNN代替
class MaskedConv2d(nn.Conv2d):
	def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
		super().__init__(*args, **kwargs)

		if mask_type not in ("A", "B"):
			raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

		self.register_buffer("mask", torch.ones_like(self.weight.data))
		_, _, h, w = self.mask.size()
		self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
		self.mask[:, :, h // 2 + 1:] = 0


class Decoder(nn.Module):
	def __init__(self, args, device, out_dim):
		super().__init__()
		self.args = args
		self.L = args.L
		self.mlp = SynthesisTransform(args.L+out_dim, args.dim_hidden, args.num_layers)
		self.context_model = MaskedConv2d(1, 2, kernel_size=5, padding=2, stride=1)
		self.laplace_conditional = LaplaceConditional(None)
		self.lower_bound_scale = LowerBound(0.11)
		self.device = device


	def get_features(self, latents, image):
		C, H, W = image.shape
		size = (H, W)
		bicubic = nn.Upsample(size=size, scale_factor=None, mode='bicubic', align_corners=True)
		y = []
		z = []
		mark = 0
		for j in range(self.L):
			H_inter = int(H / (2 ** j))
			W_inter = int(W / (2 ** j))
			lat_size = H_inter * W_inter
			y_inter = latents[mark:mark + lat_size].reshape(1, 1, H_inter, W_inter)
			# y_inter = quantize(y_inter)
			mark += lat_size
			y.append(y_inter)
			z_inter = bicubic(y_inter)
			z.append(z_inter)
		z = torch.stack(z, dim=0)
		z = z.squeeze()
		norm = nn.LayerNorm(normalized_shape=[H, W], eps=1e-05, elementwise_affine=False, device=self.device)
		z = norm(z)
		z = torch.movedim(z, 0, 2)
		return y, z

	def cul_bpp(self, y):
		bpp = 0
		for f_layer in y:
			N, _, H, W = f_layer.size()
			num_pixels = N * H * W
			ctx_params = self.context_model(f_layer)
			scales, means = ctx_params.chunk(2, 1)
			scales = self.lower_bound_scale(scales)
			_, y_likelihoods = self.laplace_conditional(f_layer, scales, means=means)
			bpp += torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
		return bpp

	def forward(self, z):
		return self.mlp(z)