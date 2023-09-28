''' Define the Transformer model '''

import numpy as np
from mindspore import nn
from mindspore.ops import operations as P, functional as F
import mindspore.ops as op
from mindspore import Tensor, Parameter


class ScaledDotProductAttention(nn.Cell):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(keep_prob=1-attn_dropout)
        self.matmul = P.BatchMatMul()
        self.transpose = P.Transpose()
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, q, k, v, mask=None):
        attn = self.matmul(q / self.temperature, self.transpose(k, (0,1,3,2)))

        if mask is not None:
            attn = F.tensor_mul((mask == 0), -1e9) + attn

        attn = self.dropout(self.softmax(attn))
        output = self.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Cell):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Dense(d_model, n_head * d_k)
        self.w_ks = nn.Dense(d_model, n_head * d_k)
        self.w_vs = nn.Dense(d_model, n_head * d_v)
        self.fc = nn.Dense(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(keep_prob=1-dropout)
        self.batch_norm = nn.BatchNorm1d(num_features=d_model)

    def construct(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = P.Reshape()(self.w_qs(q), (sz_b, len_q, n_head, d_k))
        k = P.Reshape()(self.w_ks(k), (sz_b, len_k, n_head, d_k))
        v = P.Reshape()(self.w_vs(v), (sz_b, len_v, n_head, d_v))

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = P.Transpose()(q, (0, 2, 1, 3)), P.Transpose()(k, (0, 2, 1, 3)), P.Transpose()(v, (0, 2, 1, 3))

        if mask is not None:
            mask = F.expand_dims(mask == 0)

        q , attn= self.attention(q,k,v,mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
            
        q= P.Reshape()(P.Transpose()(q, (0, 2, 1, 3)), (sz_b,len_q,-1))
        q= F.add(self.dropout(self.fc(q)),residual)
        seq_len=q.shape[1]
        q=self.batch_norm(P.Reshape()(q,(-1,self.d_model)))
        return P.Reshape()(q,(-1, seq_len, self.d_model)) ,attn


class PositionwiseFeedForward(nn.Cell):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_in = d_in
        self.w_1 = nn.Dense(d_in, d_hid) # position-wise
        self.w_2 = nn.Dense(d_hid, d_in) # position-wise
        self.batch_norm = nn.BatchNorm1d(num_features=d_in)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x):

        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        seq_len = x.shape[1]
        x = self.batch_norm(P.Reshape()(x,(-1,self.d_in)))
        return P.Reshape()(x,(-1,seq_len, self.d_in))


class EncoderLayer(nn.Cell):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def construct(self, enc_input, slf_attn_mask=None):
        enc_output , enc_slf_attn= self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output , enc_slf_attn


class Encoder(nn.Cell):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,n_layers,n_head,d_k,d_v,
            d_model,d_inner,dropout=0.,n_position=200):

        super(Encoder,self).__init__()

        self.dropout = nn.Dropout(keep_prob=1-dropout)
        self.layer_stack = nn.CellList([
            EncoderLayer(d_model,d_inner,n_head,d_k,d_v,dropout=dropout)
            for _ in range(n_layers)])

    def construct(self, src_set , src_mask , return_attns=False):

        enc_slf_attn_list = []

        enc_output = src_set
        for enc_layer in self.layer_stack:
            enc_output , enc_slf_attn= enc_layer(enc_output , slf_attn_mask=src_mask)
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)

        if return_attns:
            return enc_output , enc_slf_attn_list

        return enc_output


class Embedder(nn.Cell):
    def __init__(self, d_input, d_model):
        super(Embedder, self).__init__()
        self.conv1d = nn.Conv1d(d_input, d_model, kernel_size=1, pad_mode='valid')
        self.batch_norm = nn.BatchNorm1d(d_model)

    def construct(self, inputs):
        embeddings = self.conv1d(inputs.transpose(0, 2, 1))
        embeddings = self.batch_norm(embeddings).transpose(0, 2, 1)
        return embeddings


class Pointer(nn.Cell):
    def __init__(self, d_query, d_unit):
        super(Pointer, self).__init__()
        self.tanh = nn.Tanh()
        self.w_l = nn.Dense(d_query, d_unit, has_bias=False)
        self.v = Parameter(Tensor(np.random.uniform(-(1. / np.sqrt(d_unit)), 1. / np.sqrt(d_unit), d_unit).astype(np.float32)), requires_grad=True)

    def construct(self, edge_emb, query, mask=None):
        batch = edge_emb.shape[0]
        scores = P.ReduceSum()(self.v * self.tanh(edge_emb + self.w_l(query).unsqueeze(1)), -1)
        scores = 10. * self.tanh(scores)
        if mask is not None:
            mask = mask.reshape(batch, -1)
            scores[mask == 0] = float('-inf') + 1e-9
            t1 = op.all(mask == 0, axis=-1)
            t2 = op.where(t1 == True, Tensor([1]), Tensor([0]))
            scores[t2, 0] = 0
        return scores, t2


class Glimpse(nn.Cell):
    def __init__(self, d_model, d_unit):
        super(Glimpse, self).__init__()
        self.tanh = nn.Tanh()
        self.conv1d = nn.Conv1d(d_model, d_unit, kernel_size=1, pad_mode='valid')
        self.v = Parameter(Tensor(np.random.uniform(-(1. / np.sqrt(d_unit)), 1. / np.sqrt(d_unit), d_unit).astype(np.float32)), requires_grad=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, encs, mask=None):
        encoded = self.conv1d(encs.transpose(0, 2, 1)).transpose(0, 2, 1)
        scores = P.ReduceSum()(self.v * self.tanh(encoded), -1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attention = self.softmax(scores)
        glimpse = attention.unsqueeze(-1) * encs
        glimpse = P.ReduceSum()(glimpse, 1)
        return glimpse


if __name__ == '__main__':
    glimpse = Glimpse(128, 256)
    encs = Tensor(np.random.randn(10, 20, 128).astype(np.float32))
    a = glimpse(encs)
    print(a.shape)