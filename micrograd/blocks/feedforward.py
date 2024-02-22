import torch.nn as nn
from micrograd.layers.relu import Relu
from micrograd.layers.linear import Linear
from micrograd.layers.dropout import Dropout

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
        self.relu = Relu()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
    
    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()
    

