import mindspore as ms
import mindspore.ops as ops 
import mindspore.nn as nn


class SpatialAttentionLayer(nn.Cell):
    """
    compute spatial attention scores,chebploy graph attention
    """

    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = ms.Parameter(ms.numpy.rand(num_of_timesteps))
        self.W2 = ms.Parameter(ms.numpy.rand(in_channels, num_of_timesteps))
        self.W3 = ms.Parameter(ms.numpy.rand(in_channels))
        self.bs = ms.Parameter(ms.numpy.rand(1, num_of_vertices, num_of_vertices))
        self.Vs = ms.Parameter(ms.numpy.rand(num_of_vertices, num_of_vertices))
        self.sig = ops.Sigmoid()
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x):
        """
        Args:
            x(tensor): (B,T ,N, F_in)
        Returns:
            tensor: (B,N,N)
        """
        # x --> (b n f t)
        # x * W1 --> (B,N,F,T)(T)->(B,N,F)
        # x * W1 * W2 --> (B,N,F)(F,T)->(B,N,T)
        x=x.transpose(0, 2, 3, 1)

        lhs = ops.matmul(ops.matmul(x, self.W1), self.W2)
        # (W3 * x) ^ T --> (F)(B,N,F,T)->(B,N,T)-->(B,T,N)
        rhs = ops.matmul(self.W3, x)
        rhs = rhs.transpose(0, 2, 1)
        # x = lhs * rhs --> (B,N,T)(B,T,N) -> (B, N, N)
        product = ops.matmul(lhs, rhs)
        # S = Vs * sig(x + bias) --> (N,N)(B,N,N)->(B,N,N)
        s = ops.matmul(self.Vs, self.sig(product + self.bs))
        # softmax (B,N,N)
        s_normalized = self.softmax(s)
        return s_normalized