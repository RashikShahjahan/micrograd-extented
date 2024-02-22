from micrograd.layers.embedding import InputEmbedding
from micrograd.blocks.pos_encoding import PositionalEncoding
from micrograd.blocks.multi_head_attention import MultiHeadAttentionBlock
from micrograd.blocks.feedforward import FeedForwardBlock
from micrograd.blocks.encoder import EncoderBlock, Encoder
from micrograd.blocks.decoder import DecoderBlock, Decoder
from micrograd.blocks.projection_layer import ProjectionLayer
from micrograd.blocks.transformer import Transformer

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, nhead =8 , dim_feedforward = 2048, num_encoder_layers = 6, num_decoder_layers = 6, dropout = 0.1):
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    multi_head_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
    feed_forward = FeedForwardBlock(d_model, dim_feedforward, dropout)
    encoder_block = EncoderBlock(multi_head_attn, feed_forward, dropout)
    encoder = Encoder([encoder_block for _ in range(num_encoder_layers)])
    decoder_multi_head_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
    decoder_feed_forward = FeedForwardBlock(d_model, dim_feedforward, dropout)
    decoder_block = DecoderBlock(multi_head_attn, decoder_multi_head_attn, decoder_feed_forward, dropout)
    decoder = Decoder([decoder_block for _ in range(num_decoder_layers)])
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    return Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)