import torch

a = torch.rand(4, 1)
print(a)

b = torch.tensor([[2, 3, 4], [1, 2, 3], [123, 34, 23], [24, 53, 2]]).float()
print(b)

a = a.expand_as(b)
print(a)