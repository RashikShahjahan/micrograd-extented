from micrograd.module import Module
from micrograd.engine import Value
import random
import math
import numpy as np

class Embedding(Module):
    def __init__(self, d_model, vocab_size):

        self.weight = np.array([[Value(random.uniform(-1,1)) for _ in range(d_model)] for _ in range(vocab_size)])

    def __call__(self, x):
        return self.weight[x]
    
    def parameters(self):
        return [p for w in self.weight for p in w]
    
    def __repr__(self):
        return f"Embedding({len(self.weight)} x {len(self.weight[0])})"
    

class InputEmbedding(Module):
    def __init__(self,d_model,vocab_size):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(d_model,vocab_size)

    def __call__(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    def parameters(self):
        return self.embedding.parameters()
    
    def __repr__(self):
        return f"InputEmbedding({self.d_model},{self.vocab_size})"
    