import torch
from torch import nn 
import numpy as np


def cross_entropy(y,y_pre):
  loss=-np.sum(y*np.log(y_pre))
  return loss/float(y_pre.shape[0])

y=np.array([0,0,1]) #class #2

y_pre_good=np.array([0.1,0.1,0.8])
y_pre_bed=np.array([0.8,0.1,0.1])

l1=cross_entropy(y,y_pre_good)
l2=cross_entropy(y,y_pre_bed)

print('Loss 1:',l1)
print('Loss 2:',l2)

loss = nn.CrossEntropyLoss()

y=torch.tensor(2)

y_pre_good=torch.tensor([0.1,0.1,0.8])
y_pre_bed=torch.tensor([0.8,0.1,0.1])


l1=loss(y_pre_good,y)
l2=loss(y_pre_bed,y)

print(l1.item())
print(l2.item())