from module import Module
import numpy as np
from engine import Value
import random

class Linear(Module):
    def __init__(self, x_len, y_len):
        self.W = np.array([[Value(random.uniform(-1,1)) for _ in range(x_len)] for _ in range(y_len)])
        self.b = np.array([Value(0) for _ in range(y_len)])


    def __call__(self, x):
        y = self.w @ x + self.b
    
    def parameters(self):
        return [self.W, self.b]
    
    def __repr__(self):
        return f"Linear({self.W.shape[1]}, {self.W.shape[0]})"