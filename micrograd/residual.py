from module import Module
from dropout import Dropout
from layer_norm import LayerNorm

class ResidualConnection(Module):
    def __init__(self, dropout):
        self.dropout = Dropout(dropout)
        self.layer_norm = LayerNorm()

    def __call__(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))
    
    def __repr__(self):
        return f"ResidualConnection({self.dropout},{self.layer_norm})"