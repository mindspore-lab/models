import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
import numpy as np
class ConvLayer(nn.Cell):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  pad_mode='pad')
        self.norm = nn.BatchNorm2d(num_features=c_in, momentum=0.9)
        self.activation = nn.ELU()
        self.pad = nn.Pad(((0, 0), (0, 0), (1, 1)), "CONSTANT")
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2)

    def construct(self, x):
        x = self.downConv(x.transpose(0, 2, 1))
        x = ops.Squeeze(-1)(self.norm(ops.expand_dims(x, -1)))
        x = self.activation(x)
        x = self.maxPool(self.pad(x))
        x = x.swapaxes(1, 2)
        return x

class EncoderLayer(nn.Cell):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(p=dropout)
        self.activation = ops.ReLU() if activation == "relu" else ops.GeLU()

    def construct(self, x, attn_mask=None):
        # x [B, L, D]
        new_x = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.swapaxes(-1, 1))))
        y = self.dropout(self.conv2(y).swapaxes(-1, 1))

        return self.norm2(x+y)

class Encoder(nn.Cell):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

######################### ALLOT #############################
def ld(name):
        return ms.Tensor(np.load("../../Code_ALLOT/src/" + name + ".npy"), ms.float32)
    
class ALLOTEncoderLayer(nn.Cell):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(ALLOTEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        # self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1), pad_mode='pad', has_bias=True) # nn.Linear(d_model, d_ff)
        # self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,1), pad_mode='pad', has_bias=True) # nn.Linear(d_ff, d_model)
        self.conv1 = nn.Dense(d_model, d_ff)
        self.conv2 = nn.Dense(d_ff, d_model)
        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = ms.ops.relu if activation == "relu" else ms.ops.gelu

    def construct(self, x, attn_mask, adj_mats, **kwargs):
        # x [B, L, N, D]
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask,
            adj_mats, 
            **kwargs
        ))
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y).swapaxes(-1, 1))) # [B, D, N, L]
        y = self.dropout(self.conv2(y.swapaxes(-1, 1))) # [B, L, N, D]
        return self.norm2(x+y)

class ALLOTEncoder(nn.Cell):
    def __init__(self, layers, norm_layer=None):
        super(ALLOTEncoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer

    def construct(self, x, attn_mask, adj_mats, **kwargs):
        # x [B, L, N, D]
        for layer in self.layers:
            x = layer(x, attn_mask, adj_mats, **kwargs)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x