"""
-*- coding: utf-8 -*-
@Time    : 9/12/2023 3:52 pm
@Author  : Xiaopeng Li
@File    : features.py

"""

import mindspore.nn as nn
from ..utils.data import get_auto_embedding_dim


class SparseFeature:
    """The Feature Class for Sparse feature in MindSpore.

    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will share with embedding.
    """

    def __init__(self, name, vocab_size, embed_dim=None, shared_with=None):
        self.embed = None
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim if embed_dim is not None else get_auto_embedding_dim(vocab_size)
        self.shared_with = shared_with

    def __repr__(self):
        return f'<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = nn.Embedding(int(self.vocab_size), self.embed_dim)
        return self.embed


class DenseFeature:
    """The Feature Class for Dense feature in MindSpore.

    Args:
        name (str): feature's name.
        embed_dim (int): embedding vector's length, the value fixed `1`.
    """

    def __init__(self, name, embed_dim=1):
        self.name = name
        self.embed_dim = embed_dim

    def __repr__(self):
        return f'<DenseFeature {self.name}>'
