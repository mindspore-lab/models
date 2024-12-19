import mindspore
import numpy as np
from mindspore import nn, ops, Tensor
from mindspore.ops import functional as F
from einops import rearrange, repeat
from mindspore import Parameter

class KLLoss(nn.Cell):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.log_softmax = ops.LogSoftmax(axis=1)
        self.softmax = ops.Softmax(axis=1)

    def construct(self, pred, label):
        T = 3
        predict = self.log_softmax(pred / T)
        target_data = self.softmax(label / T) + 1e-7
        loss = T * T * ((target_data * (ops.log(target_data) - predict)).sum(axis=1).sum() / target_data.shape[0])
        return loss


class CosLoss(nn.Cell):
    def __init__(self):
        super(CosLoss, self).__init__()

    def construct(self, cos_sim):
        base_block = np.ones([2, 2])
        base_block_list = [base_block for _ in range(8)]
        label_matrix = np.block_diag(*base_block_list)
        label_matrix = Tensor(label_matrix, mindspore.float32)

        cos_loss = ops.sqrt(ops.pow(cos_sim - label_matrix, 2).sum(axis=1))
        loss = (1 / 16) * cos_loss.sum()
        return loss


class CosLoss2(nn.Cell):
    def __init__(self):
        super(CosLoss2, self).__init__()

    def construct(self, cos_sim):
        cos_sim = ops.clip_by_value(cos_sim, 0.0, None)
        base_block = np.ones([2, 2])
        base_block_list = [base_block for _ in range(8)]
        label_matrix = np.block_diag(*base_block_list)
        label_matrix = Tensor(label_matrix, mindspore.float32)

        cos_loss = ops.sqrt(ops.pow(cos_sim - label_matrix, 2).sum(axis=1))
        loss = (1 / 16) * cos_loss.sum()
        return loss


class CenterClusterLoss(nn.Cell):
    def __init__(self):
        super(CenterClusterLoss, self).__init__()

    def construct(self, x_clusters, x1_rep, x2_rep):
        x_clusters_dual = rearrange(repeat(x_clusters.expand_dims(1), 'b 1 n -> b t n', t=2), 'b t n -> (b t) n')
        dist_x1 = ops.sqrt(ops.pow(x1_rep - x_clusters_dual, 2).sum(axis=1))
        dist_x2 = ops.sqrt(ops.pow(x2_rep - x_clusters_dual, 2).sum(axis=1))
        dist = ops.concat((dist_x1, dist_x2), axis=0)
        loss_1 = dist.mean()

        dist_2 = ops.pow(x_clusters, 2).sum(axis=1, keepdims=True).expand_dims(0) + \
                 ops.pow(x_clusters, 2).sum(axis=1, keepdims=True).expand_dims(1)
        dist_2 = dist_2 - 2 * ops.matmul(x_clusters, x_clusters.T)
        dist_2 = ops.sqrt(ops.clip_by_value(dist_2, 1e-12, None))

        dist_2 = ops.clip_by_value(dist_2, 0.1, 1.0)
        ref_matrix = Tensor(np.ones((8, 8)) - np.eye(8), mindspore.float32)
        loss_2 = ops.sqrt(ops.pow(dist_2 - ref_matrix, 2).sum(axis=1)).mean()

        loss = loss_1 + loss_2
        return loss


class CosineCenterClusterLoss(nn.Cell):
    def __init__(self):
        super(CosineCenterClusterLoss, self).__init__()
        self.margin = 0.5
        self.cosine_similarity = ops.CosineSimilarity(axis=1)

    def construct(self, x_clusters, x1_rep, x2_rep):
        x_clusters_dual = rearrange(repeat(x_clusters.expand_dims(1), 'b 1 n -> b t n', t=2), 'b t n -> (b t) n')
        x_clusters_dual = x_clusters_dual.T.expand_dims(0)

        sim_x1 = self.cosine_similarity(x1_rep.expand_dims(-1), x_clusters_dual)
        sim_x2 = self.cosine_similarity(x2_rep.expand_dims(-1), x_clusters_dual)

        sim_x = ops.concat((sim_x1, sim_x2), axis=1)
        sim_x = ops.clip_by_value(sim_x, None, self.margin)

        ref_matrix = Tensor(0.5 * np.ones(sim_x.shape), mindspore.float32)

        loss_1 = ops.sqrt(ops.pow(sim_x - ref_matrix, 2).sum(axis=1)).mean()

        sim_cluster = self.cosine_similarity(x_clusters.expand_dims(-1), x_clusters.T.expand_dims(0))
        sim_cluster = ops.clip_by_value(sim_cluster, self.margin, None)

        ref_matrix = Tensor(0.5 * np.eye(8) + 0.5 * np.ones((8, 8)), mindspore.float32)
        loss_2 = ops.sqrt(ops.pow(sim_cluster - ref_matrix, 2).sum(axis=1)).mean()

        loss = loss_1 + loss_2
        return loss


class OriTripletLoss(nn.Cell):
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin)

    def construct(self, inputs, targets):
        n = inputs.shape[0]
        dist = ops.pdist(inputs, inputs.T)
        mask = ops.equal(targets.expand_dims(0), targets.expand_dims(1))
        dist_ap = ops.max(dist[mask], axis=1)
        dist_an = ops.min(dist[~mask], axis=1)
        y = ops.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        correct = ops.sum(ops.ge(dist_an, dist_ap))
        return loss, correct
