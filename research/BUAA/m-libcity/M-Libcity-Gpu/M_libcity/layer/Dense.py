import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class Dense(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super(Dense, self).__init__()
        self.layer=nn.Dense(dim_in,dim_out)
        
    def construct(self, x):
        output = self.layer(x)
        return output
        