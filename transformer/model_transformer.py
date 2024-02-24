from micrograd.layers.embedding import InputEmbedding
from micrograd.blocks.pos_encoding import PositionalEncoding
from micrograd.blocks.multi_head_attention import MultiHeadAttentionBlock
from micrograd.blocks.feedforward import FeedForwardBlock
from micrograd.blocks.projection_layer import ProjectionLayer
from micrograd.models.transformer import Transformer, Encoder, Decoder, EncoderBlock, DecoderBlock
from torch import nn


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    print(f'src_embed: {src_embed}')
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    print(f'tgt_embed: {tgt_embed}')

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    print(f'src_pos: {src_pos}')
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    print(f'tgt_pos: {tgt_pos}')
    
    # Create the encoder blocks
    encoder_blocks = []
    for i in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        print(f'encoder_self_attention_block{i}: {encoder_self_attention_block}')
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        print(f'feed_forward_block{i}: {feed_forward_block}')
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        print(f'encoder_block{i}: {encoder_block}')
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for i in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        print(f'decoder_self_attention_block{i}: {decoder_self_attention_block}')
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        print(f'decoder_cross_attention_block{i}: {decoder_cross_attention_block}')
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        print(f'feed_forward_block{i}: {feed_forward_block}')
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        print(f'decoder_block{i}: {decoder_block}')
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    print(f'encoder: {encoder}')
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    print(f'decoder: {decoder}')
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    print(f'projection_layer: {projection_layer}')
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    print(f'transformer: {transformer}')
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer



