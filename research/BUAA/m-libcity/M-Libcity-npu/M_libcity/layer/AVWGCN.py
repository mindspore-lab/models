import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer
import mindspore.numpy as numpy

# 可训练邻接矩阵+非共享权重GCN卷积
class AVWGCN(nn.Cell):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = mindspore.Parameter(initializer('normal', (embed_dim, cheb_k, dim_in, dim_out)), name='weight')
        self.bias_pool = mindspore.Parameter(initializer('Uniform', [embed_dim, dim_out]), name='bias')
        self.softmax = nn.Softmax(axis=1)
        self.relu = nn.ReLU()
        
    def construct(self,x,node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = ops.MatMul(transpose_b=True)(node_embeddings, node_embeddings)
        supports = self.relu(supports)
        supports = self.softmax(supports)
        supports = mindspore.Tensor(supports, dtype=mindspore.float32)
        support_set = [numpy.eye(node_num, dtype=mindspore.float32), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            tmp = mindspore.Tensor(ops.MatMul()(2 * supports, support_set[-1]) - support_set[-2], dtype=mindspore.float32)
            support_set.append(tmp)
        supports = ops.Stack(axis=0)(support_set)

        # weights = ops.Einsum("nd,dkio->nkio")((node_embeddings, self.weights_pool))  # N, cheb_k, dim_in, dim_out
        nshape = node_embeddings.shape
        wshape = self.weights_pool.shape
        node_embeddings_op = node_embeddings.reshape(nshape[0], nshape[1], 1, 1, 1)
        weights_pool_op = self.weights_pool.reshape(1, wshape[0], wshape[1], wshape[2], wshape[3])
        weights = node_embeddings_op * weights_pool_op
        weights = weights.sum(axis=1)
        
        bias = ops.MatMul()(node_embeddings, self.bias_pool)                       # N, dim_out
        x1 = mindspore.Tensor(x, dtype=mindspore.float32)
        # x_g = ops.Einsum("knm,bmc->bknc")((supports, x1))      # B, cheb_k, N, dim_in
        supports_op = supports.reshape(1, supports.shape[0], supports.shape[1], supports.shape[2], 1)
        x1_op = x1.reshape(x1.shape[0], 1, 1, x1.shape[1], x1.shape[2])
        x_g = supports_op * x1_op
        x_g = x_g.sum(axis=3)
    
        x_g = ops.Transpose()(x_g, (0, 2, 1, 3))

        
        weights = mindspore.Tensor(weights, dtype=mindspore.float32)

        b,n,k,i=x_g.shape
        n,k,i,o=weights.shape
        x_g=x_g.expand_dims(-1)
        weights=weights.expand_dims(0)

        x_gconv=x_g*weights
        x_gconv=x_gconv.sum(2).sum(2)
        # print(x_gconv.shape)
        x_gconv = x_gconv + bias     # b, N, dim_out
        return x_gconv