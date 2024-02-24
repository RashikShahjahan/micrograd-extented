import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, x_len, y_len, bias=True):
        super().__init__()
        self.bias = bias
        self.w = torch.rand(x_len, y_len).to('cuda' if torch.cuda.is_available() else 'cpu')

        self.b = torch.zeros(y_len).to('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, x):
        y = x @ self.w + self.b
        return y
    
    def parameters(self):
        if self.bias:
            return [self.w, self.b]
        else:
            return [self.w]
    
    def __repr__(self):
        return f"Linear({self.w.shape[0]}, {self.w.shape[1]})"