import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np

# 时空嵌入矩阵，真正的时空特征的嵌入表示
class PositionEmbedding(nn.Cell):
    def __init__(self, input_length, num_of_vertices, embedding_size, temporal=True, spatial=True, config=None):
        super(PositionEmbedding, self).__init__()
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.embedding_size = embedding_size
        self.temporal = temporal
        self.spatial = spatial
        self.temporal_emb = ms.Parameter(np.zeros((1, input_length, 1, embedding_size)))
        # shape is (1, T, 1, C)
        self.spatial_emb = ms.Parameter(np.zeros((1, 1, num_of_vertices, embedding_size)))
        # shape is (1, 1, N, C)

    def construct(self, data):
        if self.temporal:
            data += self.temporal_emb
        if self.spatial:
            data += self.spatial_emb
        return data