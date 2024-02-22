from micrograd.module import Module
import numpy as np
from micrograd.engine import Value

class Dropout(Module):
    def __init__(self, dropout):
        self.dropout = dropout

    def __call__(self, x):
        if self.dropout == 1: 
            return np.vectorize(Value)(np.zeros_like(x))
        mask = np.vectorize(Value)(np.rand(x.shape) > self.dropout).float()
        return mask * x / (1.0 - self.dropout)

    def __repr__(self):
        return f"Dropout({self.dropout})"