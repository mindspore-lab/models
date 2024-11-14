import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
class PositionwiseFeedForward(nn.Cell):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Dense(d_in, d_hid) # position-wise
        self.w_2 = nn.Dense(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm([d_in], epsilon=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.relu = ops.ReLU()

    
    def construct(self, x):

        residual = x
        #x = self.w_2(F.relu(self.w_1(x)))
        x = self.w_2(self.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x