import torch
import torch.nn.functional as F

# assume we have some tensor x with size (b, t, k)
x = torch.rand(2, 3, 4)

print(x)

raw_weights = torch.bmm(x, x.transpose(1, 2))
# - torch.bmm is a batched matrix multiplication. It
#   applies matrix multiplication over batches of
#   matrices.


print(x.transpose(2, 1))

print(raw_weights)


weights = F.softmax(raw_weights, dim=3)

print(weights)
