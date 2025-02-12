import time
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import math
from math import sqrt
import os

def decor_time(func):
    def func2(*args, **kw):
        now = time.time()
        y = func(*args, **kw)
        t = time.time() - now
        print('call <{}>, time={}'.format(func.__name__, t))
        return y
    return func2

class AutoCorrelation(nn.Cell):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        ops_mean = ops.ReduceMean()
        ops_stack = ops.Stack()
        ops_topk = ops.TopK()
        mean_value = ops_mean(ops_mean(corr, axis=1), axis=1)
        index = ops_topk(ops_mean(mean_value, axis=0), top_k)[1]
        weights = ops.stack([mean_value[:, index[i]] for i in range(top_k)], axis=-1)
        # update corr
        tmp_corr = ops.softmax(weights, axis=-1)
        # aggregation
        tmp_values = values
        zeroslike = ops.ZerosLike()
        delays_agg = zeroslike(values).float()
        for i in range(top_k):
            pattern = ms.numpy.roll(tmp_values, -int(index[i]), 0)
            delays_agg = delays_agg + pattern * \
                         (ms.numpy.tile(tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1), (1, head, channel, length)))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = ms.numpy.tile(ms.numpy.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            , (batch, head, channel, 1))
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = ops.ReduceMean()(ops.ReduceMean()(corr, axis=1), axis=1)
        weights, delay = ops.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = ops.softmax(weights, axis=-1)
        # aggregation
        tmp_values = ms.numpy.tile(values, (1, 1, 1, 2))
        delays_agg = ops.ZerosLike()(values).float()
        for i in range(top_k):
            tmp_delay = init_index + ms.numpy.tile(delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1), (1, head, channel, length))
            pattern = ops.gather_elements(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (ms.numpy.tile(tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1), (1, head, channel, length)))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = ms.numpy.tile(ms.numpy.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            , (batch, head, channel, 1)).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = ops.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = ops.softmax(weights, axis=-1)
        # aggregation
        tmp_values = ms.numpy.tile(values, (1, 1, 1, 2))
        delays_agg = ops.ZerosLike()(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = ops.gather_elements(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def construct(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = ops.ZerosLike()(queries[:, :(L - S), :]).float()
            values = ops.concat([values, zeros], axis=1)
            keys = ops.concat([keys, zeros], axis=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        rfft_net = ops.FFTWithSize(signal_ndim=1, inverse=False, real=True)
        irfft_net = ops.FFTWithSize(signal_ndim=1, inverse=True, real=True)
        trans = ops.Transpose()
        q_fft = rfft_net(trans(queries,(0, 2, 3, 1)))
        k_fft = rfft_net(trans(keys,(0, 2, 3, 1)))
        res = q_fft * ops.conj(k_fft)
        corr = irfft_net(res)
        # time delay agg
        if self.training:
            V = trans(self.time_delay_agg_training(trans(values,(0, 2, 3, 1)), corr),(0, 3, 1, 2))
        else:
            V = trans(self.time_delay_agg_inference(trans(values,(0, 2, 3, 1)), corr),(0, 3, 1, 2))

        if self.output_attention:
            return (V, trans(corr,(0, 3, 1, 2)))
        else:
            return (V, None)

def bj(a, b, eps=1e-2):
    return (ms.ops.abs(a - b) > eps).sum()

def load(name):
    print("load", name)
    return ms.Tensor(np.load(f"../FEDformer-master/{name}.npy"))

class AutoCorrelationLayer(nn.Cell):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
