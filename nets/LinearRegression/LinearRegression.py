import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class LinearRegression(nn.Module):
    def __init__(self, in_features=8):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1)
        self.input_shape = (in_features,)
        self.output_shape = (1,)

    def forward(self, x):
        return self.linear(x) 


    

     
