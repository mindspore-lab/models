import mindspore as ms
from mindspore import nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform
import numpy as np


class EmbeddingLayer(nn.Cell):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size=sum(field_dims).item(), embedding_size = embed_dim, embedding_table='xavieruniform')
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long).tolist()

    def construct(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        exp = ops.ExpandDims()
        x = x + exp(ms.numpy.full(x.shape,self.offsets),0)
        return self.embedding(x)

class MultiLayerPerceptron(nn.Cell):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super(MultiLayerPerceptron,self).__init__(auto_prefix=True)
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Dense(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(keep_prob=1-dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Dense(input_dim, 1))
        self.mlp = nn.SequentialCell(*layers)

    def construct(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)