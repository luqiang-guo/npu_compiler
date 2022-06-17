import torch
import torch.nn as nn


x = torch.randn(16, dtype=torch.float, device = torch.device('cuda:0'))
m = nn.ReLU()
y = m(x)
print(y)