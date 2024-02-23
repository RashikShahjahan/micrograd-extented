import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self,features, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.alpha = torch.ones(features).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.bias = torch.zeros(features).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        mean = torch.mean(x, axis=-1, keepdims=True)
        std = torch.std(x, axis=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
