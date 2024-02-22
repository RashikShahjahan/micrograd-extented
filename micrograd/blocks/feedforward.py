from micrograd.module import Module
from micrograd.layers.linear import Linear
from micrograd.layers.dropout import Dropout
import numpy as np

class FeedForwardBlock(Module):
    def __init__(self, d_model, d_ff, dropout):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)
        self.relu = np.vectorize(lambda x: x.relu())

    def __call__(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
    
    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()
    
    def __repr__(self):
        return f"FeedForwardBlock({self.linear1},{self.linear2},{self.dropout})"