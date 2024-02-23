import torch.nn as nn
from micrograd.blocks.residual import ResidualConnection
from micrograd.layers.layer_norm import LayerNorm

class DecoderBlock(nn.Module):
    def __init__(self,features, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(features,dropout)
        self.residual2 = ResidualConnection(features,dropout)
        self.residual3 = ResidualConnection(features,dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        return self.residual3(x, self.feed_forward)
    
    def parameters(self, recurse):
        params = list(self.self_attn.parameters(recurse)) + list(self.cross_attn.parameters(recurse)) + self.feed_forward.parameters()
        return params
    


class Decoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
    
    def parameters(self, recurse):
        return [param for layer in self.layers for param in layer.parameters(recurse)]
