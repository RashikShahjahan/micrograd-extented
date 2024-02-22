from embedding import InputEmbedding
from pos_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttentionBlock
from feedforward import FeedForward
from encoder import EncoderBlock, Encoder
from decoder import DecoderBlock, Decoder
from projection_layer import ProjectionLayer
from transformer import Transformer

def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, dropout):
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len)
    multi_head_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
    feed_forward = FeedForward(d_model, dim_feedforward, dropout)
    encoder_block = EncoderBlock(multi_head_attn, feed_forward, dropout)
    encoder = Encoder([encoder_block for _ in range(num_encoder_layers)])
    decoder_multi_head_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
    decoder_feed_forward = FeedForward(d_model, dim_feedforward, dropout)
    decoder_block = DecoderBlock(multi_head_attn, decoder_multi_head_attn, decoder_feed_forward, dropout)
    decoder = Decoder([decoder_block for _ in range(num_decoder_layers)])
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    return Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)