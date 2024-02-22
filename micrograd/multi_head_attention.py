from module import Module
from linear import Linear
from dropout import Dropout
import numpy as np

class MultiHeadAttentionBlock(Module):
    def __init__(self, d_model, n_heads, dropout):
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
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if mask:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.softmax(-1)
        if dropout:
            scores = dropout(scores)
        output = np.matmul(scores, v), scores
        return output


    
    def __call__(self, q, k, v, mask=None):
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
    
    def parameters(self):
        return self.q_linear.parameters() + self.k_linear.parameters() + self.v_linear.parameters() + self.out.parameters()
    
    def __repr__(self):
        return f"MultiHeadAttentionBlock({self.q_linear},{self.k_linear},{self.v_linear},{self.out},{self.dropout})"
