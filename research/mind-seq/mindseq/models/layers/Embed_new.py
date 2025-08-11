import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import math

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True

class PositionalEmbedding(nn.Cell):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = ops.stop_gradient(ops.zeros((max_len, d_model), ms.float32))
        cast = ops.Cast()
        expand = ops.ExpandDims()
        position = expand(cast(ms.numpy.arange(0, max_len), ms.float32), 1)
        div_term = ops.exp((cast(ms.numpy.arange(0, d_model, 2), ms.float32) * -(math.log(10000.0) / d_model)))

        self.pe[:, 0::2] = ops.Sin()(position * div_term)
        self.pe[:, 1::2] = ops.Cos()(position * div_term)
        self.pe = expand(self.pe, 0)

    def construct(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, pad_mode='pad', has_bias=False)
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv1d):
                ms.common.initializer.HeNormal(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def construct(self, x):
        trans = ops.Transpose()
        x = self.tokenConv(trans(x,(0, 2, 1))).transpose(0, 2, 1) # transpose(1, 2)
        return x


class FixedEmbedding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = ops.stop_gradient(ops.zeros((c_in, d_model), ms.float32))

        position = ms.numpy.arange(0, c_in).float().unsqueeze(1)
        div_term = (ms.numpy.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = ms.ops.sin(position * div_term)
        w[:, 1::2] = ms.ops.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = ms.Parameter(w, requires_grad=False)

    def construct(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Cell):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def construct(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Cell):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Dense(d_inp, d_model, has_bias=False)

    def construct(self, x):
        return self.embed(x)


class DataEmbedding(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)