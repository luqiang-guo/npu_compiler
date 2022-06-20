import torch
import torch.nn as nn


# x = torch.randn(16, dtype=torch.float, device = torch.device('cuda:0'))
# m = nn.ReLU()
# y = m(x)
# print(y)


x = torch.ones([2,3],  dtype=torch.uint8)
print(x/255.0)


y = torch.tensor(int("1"))
print(y)