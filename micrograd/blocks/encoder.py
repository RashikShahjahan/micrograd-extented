import torch.nn as nn
from micrograd.layers.layer_norm import LayerNorm
from micrograd.blocks.residual import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(self, features, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(features,dropout)
        self.residual2 = ResidualConnection(features,dropout)
    
    def forward(self, x, mask):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, mask))
        return self.residual2(x, self.feed_forward)
    
    def parameters(self):
        return list(self.feed_forward.parameters()) + list(self.self_attn.parameters(True))
    

 
class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    

    

   