import torch.nn as nn
import torch

from micrograd.layers.linear import Linear
from micrograd.layers.softmax import Softmax

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, d_vocab):
        super().__init__()
        self.linear = Linear(d_model, d_vocab)
        self.softmax = Softmax()

    def forward(self, x):
        return self.softmax(self.linear(x))

    def parameters(self):
        return self.linear.parameters()