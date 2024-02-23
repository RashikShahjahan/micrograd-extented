import torch.nn as nn
import torch

class Relu(nn.Module):
    def forward(self, x):
        return torch.max(x, torch.zeros_like(x)).to('cuda' if torch.cuda.is_available() else 'cpu')
    