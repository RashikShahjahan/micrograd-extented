import torch.nn as nn

class Softmax(nn.Module):
    def forward(self, x):
        out = x.exp() / x.exp().sum(-1).unsqueeze(-1)
        return out
    
        
