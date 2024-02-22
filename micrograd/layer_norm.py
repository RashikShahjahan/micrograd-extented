from module import Module
import numpy as np
from engine import Value

class LayerNorm(Module):
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.alpha = np.vectorize(Value)(np.ones(1))
        self.bias = np.vectorize(Value)(np.zeros(1))

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    def parameters(self):
        return self.alpha, self.bias
    
    def __repr__(self):
        return f"LayerNorm({self.alpha},{self.bias})"