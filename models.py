"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

class QANet(nn.Module):
    """Custom implementation of QANet model for SQuAD.

    Based on the paper:
    "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"
    by Adams Wei Yu, David Dohan, Minh-Thang Luong
    (https://arxiv.org/pdf/1804.09541.pdf)

    Follows the following high-level structure:
        - Embedding layer
        - Encoder layer
        - Attention layer
        - Model encoder layer
        - Output layer

    Args:
        char_vectors (torch.Tensor): Pre-trained char vectors.
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob=0.):
        super(QANet, self).__init__()
        self.input_embedding = layers.QANetEmbedding(
            char_vectors=char_vectors,
            word_vectors=word_vectors,
            hidden_size=hidden_size,
            drop_prob=drop_prob
        )
        self.embedding_encoder = layers.EncoderBlock(
            hidden_size=hidden_size,
            k=7,
            drop_prob=drop_prob,
            num_convs=4
        )
        self.attention = layers.BiDAFAttention(
            hidden_size=hidden_size,
            drop_prob=drop_prob
        )
        self.model_encoder = nn.ModuleList([layers.EncoderBlock(
            hidden_size=hidden_size,
            k=7,
            drop_prob=drop_prob,
            num_convs=2
        )] * 7)
        self.output = layers.QANetOutput(
            hidden_size=hidden_size
        )
        self.convs_map_c = layers.DepthwiseSeparableConvolution(500, hidden_size, 7)
        self.convs_map_q = layers.DepthwiseSeparableConvolution(500, hidden_size, 7)
        self.convs_map_att = layers.DepthwiseSeparableConvolution(400, hidden_size, 7)

    def forward(self, cc_idxs, qc_idxs, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.input_embedding(cc_idxs, cw_idxs)
        q_emb = self.input_embedding(qc_idxs, qw_idxs)
        c_emb = c_emb.permute(0, 2, 1)
        c_emb = self.convs_map_c(c_emb)
        c_emb = c_emb.permute(0, 2, 1)
        q_emb = q_emb.permute(0, 2, 1)
        q_emb = self.convs_map_q(q_emb)
        q_emb = q_emb.permute(0, 2, 1)

        c_enc = self.embedding_encoder(c_emb)
        q_enc = self.embedding_encoder(q_emb)

        att = self.attention(c_enc, q_enc, c_mask, q_mask)
        att = att.permute(0, 2, 1)
        att = self.convs_map_att(att)
        att = att.permute(0, 2, 1)

        M0 = att
        for model_encoder_block in self.model_encoder:
            M0 = model_encoder_block(M0)

        M1 = M0
        for model_encoder_block in self.model_encoder:
            M1 = model_encoder_block(M1)

        M2 = M1
        for model_encoder_block in self.model_encoder:
            M2 = model_encoder_block(M2)

        start_prob, end_prob = self.output(M0, M1, M2, c_mask)

        return start_prob, end_prob


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
