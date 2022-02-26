import torch
import torch.nn as nn

import torch.nn.functional as F

input = torch.randn(3,5, requires_grad=True)
target = torch.empty(3,dtype=torch.long).random_(5)

class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()
		self.linear1 = nn.Linear(5,5)

	def forward(self,x):
		y_hat = self.linear1(x)
		return y_hat

