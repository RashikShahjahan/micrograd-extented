import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.alpha = torch.ones(1)
        self.bias = torch.zeros(1)

    def forward(self, x):
        mean = torch.mean(x, axis=-1, keepdims=True)
        std = torch.std(x, axis=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
