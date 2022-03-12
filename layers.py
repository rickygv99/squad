"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

class QANetEmbedding(nn.Module):
    """Embedding layer used by QANet.

    Args:
        p_1 (int): Number of dimensions in pre-trained GloVe word vectors
        p_2 (int): Number of dimensions in characters
    """
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob, p_1=300, p_2=200):
        super(QANetEmbedding, self).__init__()
        self.p_1 = p_1
        self.p_2 = p_2
        self.drop_prob = drop_prob
        self.w_embed = nn.Embedding.from_pretrained(word_vectors)
        self.c_embed = nn.Embedding.from_pretrained(char_vectors)
        self.conv = nn.Conv2d(self.c_embed.embedding_dim, p_2, kernel_size=(1,5), bias=True)
        self.hwy = HighwayEncoder(2, p_1 + p_2)

    def forward(self, c, w):
        x_w = self.w_embed(w) # (batch_size, seq_len, embed_size)
        x_w = F.dropout(x_w, self.drop_prob, self.training)

        x_c = self.c_embed(c) # (batch_size, seq_len, embed_size_1, embed_size_2)
        x_c = F.dropout(x_c, self.drop_prob, self.training)
        x_c = x_c.permute(0, 3, 1, 2) # (batch_size, embed_size_2, seq_len, embed_size_1)
        x_c = self.conv(x_c)
        x_c = x_c.permute(0, 2, 1, 3) # (batch_size, seq_len, embed_size_2, embed_size_1)
        x_c, _ = torch.max(x_c, dim=3)

        x = torch.cat([x_w, x_c], dim=2) # Join on embed_size
        x = self.hwy(x)
        return x

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, k, bias=True):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=k,
            padding=k//2,
            groups=in_channels,
            bias=bias
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        k_t = k.permute(0, 2, 1)
        qk_t = torch.matmul(q, k_t) / np.sqrt(self.d_k)
        x = self.softmax(qk_t)
        x = torch.matmul(x, v)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, drop_prob, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.drop_prob = drop_prob
        self.n_heads = n_heads
        self.d_model = d_model

        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_model // self.n_heads

        self.projection_q = nn.Linear(self.d_model, self.d_model * self.d_k)
        self.projection_k = nn.Linear(self.d_model, self.d_model * self.d_k)
        self.projection_v = nn.Linear(self.d_model, self.d_model * self.d_v)
        self.projection_map = nn.Linear(self.d_model * self.d_v, self.d_model)

        self.scaled_dp_attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q, k, v):
        proj_q = self.projection_q(q)
        proj_k = self.projection_k(k)
        proj_v = self.projection_v(v)

        attention = self.scaled_dp_attention(proj_q, proj_k, proj_v)
        attention = self.projection_map(attention)

        return attention

class PositionalEncoding(nn.Module):
    def __init__(self, length, d_model):
        super(PositionalEncoding, self).__init__()
        pos = torch.arange(length)
        pos = torch.tile(pos, (d_model))

        i_2 = torch.arange(d_model)[0:d_model:2]
        i_2 = torch.repeat_interleave(i_2, 2)

        self.pe = torch.zeros((length, d_model))
        self.pe[:, 0:d_model:2] = torch.sin(pos / (10000 ** (i_2 / d_model)))[:, 0:d_model:2]
        self.pe[:, 1:d_model:2] = torch.cos(pos / (10000 ** (i_2 / d_model)))[:, 1:d_model:2]
        self.pe = nn.Parameter(self.pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:, 0:x.size(dim=1)]

        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, k, drop_prob, num_convs, length, p_1=300):
        super(EncoderBlock, self).__init__()
        self.drop_prob = drop_prob
        self.num_convs = num_convs
        self.convs = nn.ModuleList([DepthwiseSeparableConvolution(hidden_size, hidden_size, k) for i in range(num_convs)])
        self.norms_convs = nn.ModuleList([nn.LayerNorm(hidden_size) for i in range(num_convs)])
        self.norm_attention = nn.LayerNorm(hidden_size)
        self.norm_feedforward = nn.LayerNorm(hidden_size)
        self.multiheadattention = MultiHeadAttention(hidden_size, drop_prob)
        self.feedforward = nn.Linear(hidden_size, hidden_size)
        self.positionalencoding = PositionalEncoding(length, p_1)

    def forward(self, x):
        x = self.positionalencoding(x)
        for i in range(self.num_convs):
            residual = x
            x = self.norms_convs[i](x)
            x = x.permute(0, 2, 1)
            x = self.convs[i](x)
            x = x.permute(0, 2, 1)
            x = x + residual
        residual = x
        x = self.norm_attention(x)
        x = self.multiheadattention(x, x, x)
        x = x + residual
        residual = x
        x = self.norm_feedforward(x)
        x = self.feedforward(x)
        x = x + residual
        return x

class QANetOutput(nn.Module):
    """Output layer used by QANet.

    Args:
    """
    def __init__(self, hidden_size):
        super(QANetOutput, self).__init__()
        self.linear_1 = nn.Linear(hidden_size * 2, 1)
        self.linear_2 = nn.Linear(hidden_size * 2, 1)

    def forward(self, M0, M1, M2, mask):
        W_1 = torch.cat((M0, M1), 2)
        W_2 = torch.cat((M0, M2), 2)
        L_1 = self.linear_1(W_1)
        L_2 = self.linear_2(W_2)
        p_1 = masked_softmax(L_1.squeeze(), mask, -1, True)
        p_2 = masked_softmax(L_2.squeeze(), mask, -1, True)
        return p_1, p_2




# CODE BELOW THIS LINE IS PART OF ORIGINAL BASELINE CODE
# --------------------------------------------------------




class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
