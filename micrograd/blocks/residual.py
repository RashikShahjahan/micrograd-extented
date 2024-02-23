
import torch.nn as nn
from micrograd.layers.dropout import Dropout
from micrograd.layers.layer_norm import LayerNorm

class ResidualConnection(nn.Module):
    def __init__(self, features, dropout):
        super().__init__()

        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
 