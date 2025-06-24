import math
import mindspore.ops as ops
import mindspore as ms
import mindspore.nn as nn

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
        return self.pe[:, :x.shape[1]]

from mindspore.common.initializer import initializer, HeNormal
class TokenEmbedding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, pad_mode='pad')
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv1d):
                m.weight.set_data(initializer(HeNormal(mode='fan_in',nonlinearity='leaky_relu'), m.weight.shape, ms.float32))

    def construct(self, x):
        x = self.tokenConv(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        return x

class TemporalEmbedding(nn.Cell):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13
        Embed = nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def construct(self, x):
        x = x.astype(ms.int64)
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Cell):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Dense(d_inp, d_model)
    
    def construct(self, x):
        y = self.embed(x)
        return self.embed(x)

class DataEmbedding(nn.Cell):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
                                    if embed_type!='timeF' else \
                                    TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def construct(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)

class PositionalEncoding(nn.Cell):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
        return self.pe[:, :x.shape[1]].unsqueeze(-2)    


class TemporalEncoding(nn.Cell):
    def __init__(self, d_model, dropout=0.1):
        super(TemporalEncoding, self).__init__()
        self.emb = nn.Dense(5, d_model)
    
    def construct(self, x):
        return self.emb(x)

class SpatialEncoding(nn.Cell):
    def __init__(self, c_in, d_model):
        super(SpatialEncoding, self).__init__()
        
        self.SE = nn.Dense(c_in, d_model)

    def construct(self, x):
        x = self.SE(x)

        return x