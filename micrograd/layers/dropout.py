import torch.nn as nn
import torch

class Dropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()

        self.dropout = dropout

    def forward(self, x):
        if self.dropout == 1: 
            return torch.zeros_like(x)
        mask = (torch.rand(x.shape) > self.dropout).float()
        return mask * x / (1.0 - self.dropout)

