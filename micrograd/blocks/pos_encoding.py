from micrograd.layers.dropout import Dropout
from micrograd.module import Module
import numpy as np


class PositionalEncoding(Module):
    def __init__(self, d_model, seq_len, dropout):
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = Dropout(dropout)

        pe = np.zeros((seq_len, d_model))
        position = np.arange(0, seq_len).reshape(-1, 1)

        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        pe = pe[np.newaxis, :, :]

        self.pe = pe

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    
    def __repr__(self):
        return f"PositionalEncoding({self.d_model},{self.seq_len},{self.dropout})"



