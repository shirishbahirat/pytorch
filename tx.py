import torch
import torch.nn as nn


# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print('Input CL',input)
target = torch.empty(3, dtype=torch.long).random_(5)
print('Target CL',target)
output = loss(input, target)
output.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
print('Input SM',input)
target = torch.randn(3, 5).softmax(dim=1)
print('Target SM',target)
output = loss(input, target)
output.backward()
print(output)
