import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn


dataset = MNIST(root='data/', download=True)

print(len(dataset))

test_dataset = MNIST(root='data/', train=False)
print(len(test_dataset))

'''
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)
plt.show()
'''


dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label)

print(img_tensor[0, 10:15, 10:15])
print(torch.max(img_tensor), torch.min(img_tensor))

train_ds, val_ds = random_split(dataset, [50000, 10000])

'''
plt.imshow(img_tensor[0, 0:28, 0:28], cmap='gray')
plt.show()
'''

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28 * 28
num_classes = 10

model = nn.Linear(input_size, num_classes)

print(model.weight.shape)
print(model.weight)


print(model.bias.shape)
print(model.bias)


for images, labels in train_loader:
    print(labels)
    print(images.shape)
    #outputs = model(images)
    # print(outputs)
    # break


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out


model = MnistModel()


for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)
