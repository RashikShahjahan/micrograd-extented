from module import Module

class Transformer(Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer):
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

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters() + self.src_embed.parameters() + self.tgt_embed.parameters() + self.src_pos.parameters() + self.tgt_pos.parameters() + self.projection_layer.parameters()
    
    def __repr__(self):
        return f"Transformer({self.encoder},{self.decoder},{self.src_embed},{self.tgt_embed},{self.src_pos},{self.tgt_pos},{self.projection_layer})"
    

