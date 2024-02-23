import math
import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = torch.rand(vocab_size,d_model).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        return self.weight[x]
    
    def parameters(self):
        return [self.weight]
    
  

class InputEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    def parameters(self):
        return self.embedding.parameters()
    

    