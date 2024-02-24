import torch.nn as nn
from micrograd.layers.layer_norm import LayerNorm
from micrograd.blocks.residual import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(self, features, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attention_block = self_attn
        self.feed_forward_block = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

 
class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

    

    

   