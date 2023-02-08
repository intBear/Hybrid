import torch
input_dim = 2
out_dim = 0
max_freq_log2 = 9
num_freq = 4
H = 32
W = 32

bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freq)
out_dim += bands.shape[0] * input_dim * 2
cords = torch.rand(1024, 2)
N = cords.shape[0]
winded = (cords[:, None] * bands[None, :, None]).reshape(N, cords.shape[1] * num_freq)
encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
encoded = encoded.reshape(H, W, out_dim)
z = torch.randn(8, 8, 7)
z = torch.cat((z,encoded), 2)
print(z.shape)
