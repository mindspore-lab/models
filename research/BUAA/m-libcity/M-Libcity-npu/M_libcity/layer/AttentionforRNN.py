import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as np
import mindspore.nn as nn

class RnnAttnOutput(nn.Cell):
    """ Attention 注意力机制模块, 对 Rnn 中间层输出做加权平均. """

    def __init__(self, hidden_size, output_feats):
        """ 初始化.
        Args:
            hidden_size (int): 中间层输出向量的大小
        """
        super(RnnAttnOutput, self).__init__()
        self.sqrt_rec_size =  ms.Tensor(1., dtype=ms.float32) / ms.Tensor(np.sqrt(hidden_size), dtype=ms.float32)
        self.output_feats=output_feats
        # context vector
        hidden_size = int(hidden_size)
        self.u = nn.Dense(hidden_size,output_feats, has_bias=False)
        self.softmax = nn.Softmax(axis=2)
        self.batmatmul = ops.BatchMatMul()

    def construct(self, x):
        """ 前向传播.
        Args:
            x (tensor.Tensor): shape (batch, seq_len, hidden_size) or (batch,T,n,hidden_size)中间层输出序列
        Returns:
            (tensor.Tensor): shape (batch, o,size) or (batch,o,n,hidden_size)
        """
        if len(x.shape) == 4:
            batch = x.shape[0]
            nodes = x.shape[2]
            x = x.reshape(-1, nodes, x.shape[3])
        w = self.u(x) * self.sqrt_rec_size
        w = w.transpose(0, 2, 1)
        w = self.softmax(w)  # batch_size * o *seq_len
        c = self.batmatmul(w,x)
        if len(x.shape) == 4:
            c = c.reshape(batch, self.output_feats,nodes, -1)
        return c