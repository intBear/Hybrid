import torch
from torch.nn.parameter import Parameter

from torch import optim
model = Parameter(torch.randn(16, requires_grad=True))


print(model)
