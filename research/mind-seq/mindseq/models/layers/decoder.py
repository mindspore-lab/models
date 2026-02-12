import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops

class DecoderLayer(nn.Cell):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])
        self.norm3 = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(p=dropout)
        self.activation = ops.ReLU() if activation == "relu" else ops.GeLU()

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention( 
            x, x, x,
            attn_mask=x_mask
        ))
        
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        ))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(0, 2, 1))))
        y = self.dropout(self.conv2(y).transpose(0, 2, 1))

        return self.norm3(x+y)

class Decoder(nn.Cell):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)

        return x

######################## ALLOT ######################
import mindspore as ms
import mindspore.nn as nn
class ALLOTDecoderLayer(nn.Cell):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(ALLOTDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,1))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,1))
        self.norm1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.norm3 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = ms.ops.relu if activation == "relu" else ms.ops.gelu

    def construct(self, x, cross, x_mask, cross_mask, adj_mats, **kwargs):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            x_mask,
            adj_mats, 
            **kwargs
        ))
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            cross, x, x,
            # x, cross, cross,
            cross_mask,
            adj_mats, 
            **kwargs
        ))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class ALLOTDecoder(nn.Cell):
    def __init__(self, layers, norm_layer=None):
        super(ALLOTDecoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer

    def construct(self, x, cross, x_mask, cross_mask, adj_mats, **kwargs):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask, adj_mats, **kwargs)

        if self.norm is not None:
            x = self.norm(x)

        return x