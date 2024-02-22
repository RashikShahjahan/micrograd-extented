from micrograd.module import Module
from micrograd.layers.layer_norm import LayerNorm
from micrograd.blocks.residual import ResidualConnection


class EncoderBlock(Module):
    def __init__(self, self_attn, feed_forward, dropout):
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
    
    def __call__(self, x, mask):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.residual2(x, self.feed_forward)
    
    def parameters(self):
        return self.self_attn.parameters() + self.feed_forward.parameters()
    
    def __repr__(self):
        return f"EncoderBlock({self.self_attn},{self.feed_forward},{self.residual1},{self.residual2})"
    
class Encoder(Module):
    def __init__(self, layers):
        self.layers = layers
        self.norm = LayerNorm()

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] + self.norm.parameters()
    
    def __repr__(self):
        return f"Encoder({self.layers},{self.norm})"
    
   