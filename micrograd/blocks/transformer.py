import torch.nn as nn

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
    
    def decode(self, tgt, encoder_out, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.project(self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask))
    
    def parameters(self):
        params = self.encoder.parameters()+ list(self.decoder.parameters(True))+self.src_embed.parameters()+self.tgt_embed.parameters()+self.projection_layer.parameters()
        print(f'Transformer{list(params)}')
        return params