# funcs_ms.py
# Diagonal Composition function, Decompose is removed as it's not needed for inference.
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import Uniform, initializer

class Compose(nn.Cell):
    def __init__(self, embedding_size, channel_size):
        super(Compose, self).__init__()
        self.E = embedding_size
        self.c = channel_size
        self.e = int(self.E / self.c)
        # Initialize parameters directly in MindSpore way
        '''
        self.comp_l = Parameter(initializer(Uniform(scale=0.3, loc=0.6), (self.e,)), name="comp_l")
        self.comp_r = Parameter(initializer(Uniform(scale=0.3, loc=0.6), (self.e,)), name="comp_r")
        self.cb = Parameter(initializer('zeros', (self.e,)), name="cb")
        self.dropout = nn.Dropout(p=0.1)
        '''
        # [FIXED] Correct parameter initialization for MindSpore's Uniform API.
        # The original PyTorch code used uniform_(0.6, 0.9).
        # We replicate this by generating from U(0, 0.3) and adding an offset of 0.6.
        # For comp_l:
        init_data_l = initializer(Uniform(scale=0.3), (self.e,)) + 0.6
        self.comp_l = Parameter(init_data_l, name="comp_l")

        # For comp_r:
        init_data_r = initializer(Uniform(scale=0.3), (self.e,)) + 0.6
        self.comp_r = Parameter(init_data_r, name="comp_r")

        self.cb = Parameter(initializer('zeros', (self.e,)), name="cb")
        self.dropout = nn.Dropout(p=0.1)

    def construct(self, in_feats, words=False):
        """
        Note: inplace=True for dropout is deprecated in MindSpore.
        The dropout operation is applied functionally.
        """
        if words:
            l_c = in_feats[0] * ops.sigmoid(self.comp_l)
            r_c = in_feats[1] * ops.sigmoid(self.comp_r)
            return (l_c + r_c + self.cb).view(-1, self.E)

        # N, children, _, _ = in_feats.shape
        # assert children == 2, "Expected to have only 2 children"
        t_in = ops.transpose(in_feats, (1, 0, 2, 3))
        l_c = t_in[0] * ops.sigmoid(self.comp_l)
        r_c = t_in[1] * ops.sigmoid(self.comp_r)
        
        composed = (l_c + r_c) + self.cb
        composed = self.dropout(composed)
        return composed.view(-1, self.E)