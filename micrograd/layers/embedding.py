import math
import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.weight = torch.rand(d_model,vocab_size)

    def forward(self, x):
        return self.weight @ x
    
    def parameters(self):
        return [self.weight]
    
  

class InputEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(d_model,vocab_size)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    def parameters(self):
        return self.embedding.parameters()
    

    