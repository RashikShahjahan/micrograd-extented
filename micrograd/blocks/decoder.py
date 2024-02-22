from micrograd.module import Module
from micrograd.blocks.residual import ResidualConnection
from micrograd.layers.layer_norm import LayerNorm

class DecoderBlock(Module):
    def __init__(self, self_attn, cross_attn, feed_forward, dropout):
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.residual1 = ResidualConnection(dropout)
        self.residual2 = ResidualConnection(dropout)
        self.residual3 = ResidualConnection(dropout)

    def __call__(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual2(x, lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        return self.residual3(x, self.feed_forward)
    
    def parameters(self):
        return self.self_attn.parameters() + self.cross_attn.parameters() + self.feed_forward.parameters()
    
    def __repr__(self):
        return f"DecoderBlock({self.self_attn},{self.cross_attn},{self.feed_forward},{self.residual1},{self.residual2},{self.residual3})"

class Decoder(Module):
    def __init__(self, layers):
        self.layers = layers
        self.norm = LayerNorm()

    def __call__(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)