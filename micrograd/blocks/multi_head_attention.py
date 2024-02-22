import torch.nn as nn
import torch
import math
from micrograd.layers.linear import Linear
from micrograd.layers.dropout import Dropout
from micrograd.layers.softmax import Softmax

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.q_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.out = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = Softmax()(scores)
        if dropout:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output, scores

    
    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]
        q = self.q_linear(q).reshape(bs, -1, self.n_heads, self.d_k)
        k = self.k_linear(k).reshape(bs, -1, self.n_heads, self.d_k)
        v = self.v_linear(v).reshape(bs, -1, self.n_heads, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x, self.scores = MultiHeadAttentionBlock.attention(q, k, v, mask, self.dropout)
        x = x.transpose(1, 2).reshape(bs, -1, self.d_model)
        return self.out(x)
    
    def parameters(self, recurse):
        return super().parameters(recurse)
        
    