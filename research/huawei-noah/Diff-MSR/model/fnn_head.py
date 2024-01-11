import mindspore

from layer import FeaturesEmbedding, MultiLayerPerceptron

class FactorizationSupportedNeuralNetworkModel_head(mindspore.nn.Cell):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        W Zhang, et al. Deep Learning over Multi-field Categorical Data - A Case Study on User Response Prediction, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        #self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def construct(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = x
        x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return mindspore.ops.sigmoid(x.squeeze(1))
