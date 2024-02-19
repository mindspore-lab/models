from mindspore import nn
import mindspore as ms
import mindspore.ops as ops
import numpy as np
from .layers import EmbeddingLayer, MultiLayerPerceptron


class AITMModel(nn.Cell):
    """
    A pytorch implementation of Adaptive Information Transfer Multi-task Model.

    Reference:
        Xi, Dongbo, et al. Modeling the sequential dependence among audience multi-step conversions with multi-task learning in targeted display advertising. KDD 2021.
    """

    def __init__(self, categorical_field_dims, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super(AITMModel, self).__init__(auto_prefix=True)
        self.emb = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.embed_output_dim = len(categorical_field_dims) * embed_dim
        self.task_num = task_num
        self.hidden_dim = bottom_mlp_dims[-1]

        self.g = nn.CellList([nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1]) for i in range(task_num - 1)])
        self.h1 = nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h2 = nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1])
        self.h3 = nn.Dense(bottom_mlp_dims[-1], bottom_mlp_dims[-1])

        self.bottom = nn.CellList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = nn.CellList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.sigmoid = ops.Sigmoid()
        self.sum = ops.ReduceSum()
        self.keepsum = ops.ReduceSum(True)
        self.cast = ops.Cast()


    def construct(self, categorical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.emb(categorical_x)
        categorical_emb = self.duplicate(categorical_emb)
        emb = [categorical_emb[i].view(-1, self.embed_output_dim) for i in range(self.task_num)]
        fea = [self.bottom[i](emb[i]) for i in range(self.task_num)]

        for i in range(1, self.task_num):
            p = ops.expand_dims(self.g[i - 1](fea[i - 1]),1)
            q = ops.expand_dims(fea[i],1)
            x = ops.concat([p, q], axis = 1)
            V = self.h1(x)
            K = self.h2(x)
            Q = self.h3(x)
            fea[i] = self.cast(self.sum(ops.softmax(self.keepsum(K * Q, 2) / ms.Tensor(np.sqrt(self.hidden_dim)), axis=1) * V, 1),ms.float32)


        results = [self.sigmoid(ops.squeeze(self.tower[i](fea[i]),1)) for i in range(self.task_num)]
        return results

    def duplicate(self,x):
        if len(x) != (self.task_num+1): # Don't set batch_size as two  
            return [x] * (self.task_num+1)
        return x

class ESMMModel(nn.Cell):
    def __init__(self, categorical_field_dims, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, dropout):
        super(ESMMModel, self).__init__(auto_prefix=True)
        self.emb = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.embed_output_dim = len(categorical_field_dims) * embed_dim
        self.task_num = task_num
        self.hidden_dim = bottom_mlp_dims[-1]
        self.bottom = nn.CellList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(task_num)])
        self.tower = nn.CellList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.sigmoid = ops.Sigmoid()

    def construct(self, categorical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.emb(categorical_x)
        categorical_emb = self.duplicate(categorical_emb)
        emb = [categorical_emb[i].view(-1, self.embed_output_dim) for i in range(self.task_num)]
        fea = [self.bottom[i](emb[i]) for i in range(self.task_num)]
        results = [self.sigmoid(self.tower[i](fea[i]).squeeze(1)) for i in range(self.task_num)]
        for i in range(1,self.task_num):
            results[i] = results[i]*results[i-1]
        return results

    def duplicate(self,x):
        if len(x) != (self.task_num+1): # Don't set batch_size as two  
            return [x] * (self.task_num+1)
        return x