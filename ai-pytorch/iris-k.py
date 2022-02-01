import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./ai-pytorch/data/kagal/train.csv")


print(dataset.head())
