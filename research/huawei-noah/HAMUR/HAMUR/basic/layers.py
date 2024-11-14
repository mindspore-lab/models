"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 4:06 pm
@Author  : Xiaopeng Li
@File    : layers.py

"""
import mindspore
import numpy as np
from mindspore import Parameter, Tensor, ParameterTuple, ops
from .features import DenseFeature, SparseFeature
from ..basic.activation import activation_layer
import mindspore.nn as nn


class EmbeddingLayer(nn.Cell):
    """General Embedding Layer.

    [Similar docstring as in PyTorch]

    """

    def __init__(self, features):
        super(EmbeddingLayer, self).__init__()
        self.features = features
        self.embed_dict = nn.CellDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  # exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with is None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def construct(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                sparse_exists = True
                embed = self.embed_dict[fea.shared_with if fea.shared_with is not None else fea.name]
                emb = embed(x[fea.name].astype(mindspore.int32)).unsqueeze(1)
                sparse_emb.append(emb)

            elif isinstance(fea, DenseFeature):
                dense_exists = True
                dense_values.append(x[fea.name].astype(mindspore.float32).unsqueeze(1))

        if dense_exists:
            dense_values = mindspore.ops.Concat(1)(dense_values)
        if sparse_exists:
            sparse_emb = mindspore.ops.Concat(1)(sparse_emb)

        if squeeze_dim:
            if dense_exists and not sparse_exists:
                # Only dense features
                output = dense_values
            elif not dense_exists and sparse_exists:
                # Only sparse features, flatten the embedding
                output = sparse_emb.view(sparse_emb.shape[0], -1)  # Equivalent to PyTorch's flatten(start_dim=1)
            elif dense_exists and sparse_exists:
                # Both dense and sparse features
                flattened_sparse = sparse_emb.view(sparse_emb.shape[0], -1)
                output = mindspore.ops.Concat(1)((flattened_sparse, dense_values))
            else:
                raise ValueError("The input features cannot be empty")
        else:
            if sparse_exists:
                # Keeping the original shape for sparse features
                output = sparse_emb
            else:
                raise ValueError("Expected SparseFeatures in feature list, got {}".format(features))

        return output


class LR(nn.Cell):
    """Logistic Regression Module for MindSpore. It applies a linear transformation
    and optionally a sigmoid function to the input feature.

    Args:
        input_dim (int): Input size of the Dense layer.
        sigmoid (bool): Whether to apply a sigmoid function before output.

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)`
    """

    def __init__(self, input_dim, sigmoid=False):
        super(LR, self).__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Dense(input_dim, 1)

    def construct(self, x):
        if self.sigmoid:
            return ops.Sigmoid(self.fc(x))
        else:
            return self.fc(x)


class MLP(nn.Cell):
    """
    Multi Layer Perceptron Module for MindSpore. It's a commonly used module for
    learning features, with BatchNorm1d, Activation, and Dropout applied to each Dense layer.

    Args:
        input_dim (int): Input size of the first Dense layer.
        output_layer (bool): Whether this MLP module is the output layer.
        If True, appends a Dense layer with output size 1.
        dims (list): Output sizes of Dense layers (default=[]).
        dropout (float): Probability of an element to be zeroed (default=0.5).
        activation (str): Activation function, supports ['sigmoid', 'relu', 'dice', 'softmax']
        (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0.0, activation="relu"):
        super(MLP, self).__init__()
        if dims is None:
            dims = []

        self.mlp = nn.SequentialCell()
        for i_dim in dims:
            self.mlp.append(nn.Dense(input_dim, i_dim))
            self.mlp.append(nn.BatchNorm1d(i_dim))
            self.mlp.append(activation_layer(activation))  # Convert to MindSpore's activation layer
            self.mlp.append(nn.Dropout(keep_prob=0.99-dropout))
            input_dim = i_dim

        if output_layer:
            self.mlp.append(nn.Dense(input_dim, 1))

    def construct(self, x):
        return self.mlp(x)


class CrossNetwork(nn.Cell):
    """CrossNetwork as mentioned in the DCN paper, adapted for MindSpore.

    Args:
        input_dim (int): Input dimension of the input tensor.
        num_layers (int): Number of cross layers.

    Shape:
        - Input: `(batch_size, *)`
        - Output: `(batch_size, *)`
    """

    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.w = nn.CellList([nn.Dense(input_dim, 1, has_bias=False, weight_init='normal') for _ in range(num_layers)])
        self.b = ParameterTuple(
            (Parameter(Tensor(np.zeros((input_dim,)), mindspore.float32), name="b"+str(i)) for i in range(num_layers)))

    def construct(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x
