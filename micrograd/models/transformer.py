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


class DecoderBlock(nn.Module):
    def __init__(self,features, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x



class Decoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
    

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self,  encoder_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
