import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, x_len, y_len):
        super().__init__()

        self.w = torch.rand(x_len,y_len)
        self.b = torch.zeros(y_len)


    def forward(self, x):
        y = self.w @ x + self.b
        return y
    
    def parameters(self):
        return [self.w, self.b]