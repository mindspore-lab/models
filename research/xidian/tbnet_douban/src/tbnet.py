# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TB-Net Model."""

from mindspore import nn
from mindspore import ParameterTuple
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore.common.dtype as mstype
from mindspore.common.api import ms_function
from mindspore.common.tensor import Tensor
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from .embedding import EmbeddingMatrix


_grad_scale = C.MultitypeFuncGraph("grad_scale")
op_mul = P.Mul()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    """Get grad with scale."""
    return op_mul(grad, F.cast(scale, F.dtype(grad)))


class TBNet(nn.Cell):
    """
    TB-Net model architecture.

    Args:
        num_items (int): Number of distinct items.
        num_references (int): Number of distinct reference objects.
        num_relations (int): Number of distinct relations.
        embedding_dim (int): Dimensions of the item, reference and relation embedding vectors.

    Inputs:
        item (Tensor): Candidate item IDs, int Tensor in shape of [batch size, ].
        rl1 (Tensor): item-reference relation IDs, int Tensor in shape of [batch size, per-item paths].
        ref (Tensor): Reference object IDs, int Tensor in shape of [batch size, per-item paths].
        rl2 (Tensor): reference-hist_item relation IDs, int Tensor in shape of [batch size, per-item paths].
        hist_item (Tensor): Historical item IDs, int Tensor in shape of [batch size, per-item paths].

    Outputs:
        scores (Tensor): Item recommendation scores, float Tensor in shape of [batch size, ]
        importances (Tensor): Relation paths' importance [0.0, 1.0], float Tensor in shape of
            [batch size, per-item paths].
        item_embs (Tensor): Candidate item embeddings, float Tensor in shape of [batch size, embedding dim]
        rl1_embs (Tensor): item-reference relation embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim, embedding dim].
        ref_embs (Tensor): Reference object embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim].
        rl2_embs (Tensor): reference-hist_item relation embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim, embedding dim].
        hist_item_embs (Tensor): Historical item embeddings, float Tensor in shape of
            [batch size, per-item paths, embedding dim].

    Supported Platforms:
        ``GPU``
    """
    def __init__(self, num_items, num_references, num_relations, embedding_dim):
        super(TBNet, self).__init__()

        self.num_entities = num_items + num_references + 1  # add dummy one for the unseen entities
        self.num_items = num_items
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.matmul = C.matmul
        self.sigmoid = P.Sigmoid()
        embedding_initializer = "normal"

        self.entity_emb_matrix = EmbeddingMatrix(self.num_entities,
                                                 self.embedding_dim,
                                                 embedding_table=embedding_initializer)
        self.relation_emb_matrix = EmbeddingMatrix(self.num_relations,
                                                   embedding_size=(self.embedding_dim, self.embedding_dim),
                                                   embedding_table=embedding_initializer)

        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(3)
        self.abs = P.Abs()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.softmax = nn.Softmax()

    def construct(self, item, rl1, ref, rl2, hist_item):
        """TB-Net main computation process."""

        item_embs = self.entity_emb_matrix(item)
        rl1_embs = self.relation_emb_matrix(rl1)
        ref_embs = self.entity_emb_matrix(ref)
        rl2_embs = self.relation_emb_matrix(rl2)
        hist_item_embs = self.entity_emb_matrix(hist_item)

        responses, importances = self._key_pathing(item_embs,
                                                   rl1_embs,
                                                   ref_embs,
                                                   rl2_embs,
                                                   hist_item_embs)

        scores = P.Squeeze()(self._predict(item_embs, responses))

        return scores, importances, item_embs, rl1_embs, ref_embs, rl2_embs, hist_item_embs

    def _key_pathing(self, item_embs, rl1_embs, ref_embs, rl2_embs, hist_item_embs):
        """
        Compute the response and path probability using item and entity embedding.
        Path structure: (candidate item, relation1, reference, relation2, historical item).

        Args:
            item_embs (Tensor): candidate item embeddings, float Tensor in shape of [batch size, dim].
            rl1_embs (Tensor): relation1 embeddings, float Tensor in shape of
                [batch size, per-item paths, dim, dim].
            ref_embs (Tensor): reference embeddings, float Tensor in shape of [batch size, per-item paths, dim].
            rl2_embs (Tensor): relation2 embeddings, float Tensor in shape of
                [batch size, per-item paths, dim, dim].
            hist_item_embs (Tensor): historical item embeddings, float Tensor in shape of
                [batch size, per-item paths, dim].

        Returns:
            responses (Tensor): user's response towards middle entity, float Tensor in shape of [batch size, dim].
            importances (Tensor): path probability/importance, float Tensor in shape of
                [batch size, per-item paths].
        """

        hist_item_e_4d = self.expand_dims(hist_item_embs, 3)
        mul_r2_hist = self.squeeze(self.matmul(rl2_embs, hist_item_e_4d))
        # path_right shape: [batch size, per-item paths, dim]
        path_right = self.abs(mul_r2_hist + self.reduce_sum(rl2_embs, 2))

        item_emb_3d = self.expand_dims(item_embs, 2)
        mul_r1_item = self.squeeze(self.matmul(rl1_embs, self.expand_dims(item_emb_3d, 1)))
        path_left = self.abs(mul_r1_item + self.reduce_sum(rl1_embs, 2))
        # path_left shape: [batch size, dim, per-item paths]
        path_left = self.transpose(path_left, (0, 2, 1))

        probs = self.reduce_sum(self.matmul(path_right, path_left), 2)
        # importances shape: [batch size, per-item paths]
        importances = self.softmax(probs)

        probs_3d = self.expand_dims(importances, 2)
        # response shape: [batch size, dim]
        responses = self.reduce_sum(ref_embs * probs_3d, 1)

        return responses, importances

    def _predict(self, item_embs, responses):
        """Predict recommendation scores."""
        scores = self.reduce_sum(item_embs * responses, 1)
        return scores


class NetWithLossCell(nn.Cell):
    """
    Network with loss function.

    Args:
        network (Cell): TBNet.
        kge_weight (float): Weight of the KG Embedding loss term.
        node_weight (float): Weight of the node loss term (default=0.002).
        l2_weight (float): Weight of the L2 regularization term (default=1e-7).

    Inputs:
        item (Tensor): Candidate item IDs, int Tensor in shape of [batch size, ].
        rl1 (Tensor): item-reference relation IDs, int Tensor in shape of [batch size, per-item paths].
        ref (Tensor): Reference object IDs, int Tensor in shape of [batch size, per-item paths].
        rl2 (Tensor): reference-hist_item relation IDs, int Tensor in shape of [batch size, per-item paths].
        hist_item (Tensor): Historical item IDs, int Tensor in shape of [batch size, per-item paths].

    Outputs:
        Tensor, the Loss value.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, network, kge_weight, node_weight=0.002, l2_weight=1e-7):
        super(NetWithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.matmul = C.matmul
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(3)
        self.abs = P.Abs()
        self.maximum = P.Maximum()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.sigmoid = P.Sigmoid()

        self.kge_weight = kge_weight
        self.node_weight = node_weight
        self.l2_weight = l2_weight

    def construct(self, item, rl1, ref, rl2, hist_item, label):
        """Forward."""
        scores, _, item_embs, rl1_embs, ref_embs, rl2_embs, hist_item_embs = \
            self.network(item, rl1, ref, rl2, hist_item)
        loss = self._loss_fun(item_embs, rl1_embs, ref_embs, rl2_embs, hist_item_embs, scores, label)

        return loss

    def _loss_fun(self, item_embs, rl1_embs, ref_embs, rl2_embs, hist_item_embs, scores, labels):
        """Loss function definition."""
        pred_loss = self.reduce_mean(self.loss(scores, labels))

        item_emb_3d = self.expand_dims(item_embs, 2)
        item_emb_4d = self.expand_dims(item_emb_3d, 1)

        mul_r1_item = self.squeeze(self.matmul(rl1_embs, item_emb_4d))

        hist_item_e_4d = self.expand_dims(hist_item_embs, 3)
        mul_r2_hist = self.squeeze(self.matmul(rl2_embs, hist_item_e_4d))

        relation1_3d = self.reduce_sum(rl1_embs, 2)
        relation2_3d = self.reduce_sum(rl2_embs, 2)

        path_left = self.reduce_sum(self.abs(mul_r1_item + relation1_3d), 2)
        path_right = self.reduce_sum(self.abs(mul_r2_hist + relation2_3d), 2)

        transr_loss = self.reduce_sum(self.maximum(self.abs(path_left - path_right), 0))
        transr_loss = self.reduce_mean(self.sigmoid(transr_loss))

        mid_entity_emb_4d = self.expand_dims(ref_embs, 3)
        mul_r2_mid = self.squeeze(self.matmul(rl2_embs, mid_entity_emb_4d))
        path_r2_mid = self.abs(mul_r2_mid + relation2_3d)

        node_loss = self.reduce_sum(self.maximum(mul_r2_hist - path_r2_mid, 0))
        node_loss = self.reduce_mean(self.sigmoid(node_loss))

        l2_loss = self.reduce_mean(self.reduce_sum(rl1_embs * rl1_embs))
        l2_loss += self.reduce_mean(self.reduce_sum(ref_embs * ref_embs))
        l2_loss += self.reduce_mean(self.reduce_sum(rl2_embs * rl2_embs))
        l2_loss += self.reduce_mean(self.reduce_sum(hist_item_embs * hist_item_embs))

        transr_loss = self.kge_weight * transr_loss
        node_loss = self.node_weight * node_loss

        l2_loss = self.l2_weight * l2_loss

        loss = pred_loss + transr_loss + node_loss + l2_loss

        return loss


class TrainStepWrapCell(nn.Cell):
    """
    Training step wrapper.

    Args:
        network (Cell): Loss network.
        lr (float): Learning rate.
        sens (float): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        item (Tensor): Candidate item IDs, int Tensor in shape of [batch size, ].
        rl1 (Tensor): item-reference relation IDs, int Tensor in shape of [batch size, per-item paths].
        ref (Tensor): Reference object IDs, int Tensor in shape of [batch size, per-item paths].
        rl2 (Tensor): reference-hist_item relation IDs, int Tensor in shape of [batch size, per-item paths].
        hist_item (Tensor): Historical item IDs, int Tensor in shape of [batch size, per-item paths].

    Outputs:
        Tensor, the Loss value.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, network, lr, sens=1):
        super(TrainStepWrapCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())
        self.lr = lr
        self.optimizer = nn.Adam(self.weights,
                                 learning_rate=self.lr,
                                 beta1=0.9,
                                 beta2=0.999,
                                 eps=1e-8)

        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reciprocal_sens = Tensor(1.0 / self.sens, mstype.float32)
        self.enable_tuple_broaden = True

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, item, rl1, ref, rl2, hist_item, label):
        """Forward."""
        weights = self.weights
        loss = self.network(item, rl1, ref, rl2, hist_item, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(item, rl1, ref, rl2, hist_item, label, sens)
        grads = self.scale_grads(grads)

        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads))

    @ms_function
    def scale_grads(self, grads):
        if self.sens != 1.0:
            return self.hyper_map(F.partial(_grad_scale, self.reciprocal_sens), grads)
        return grads


class EvalNet(nn.Cell):
    """
    Evaluation Network.

    Args:
        network (Cell): TBNet.

    Inputs:
        item (Tensor): Candidate item IDs, int Tensor in shape of [batch size, ].
        rl1 (Tensor): item-reference relation IDs, int Tensor in shape of [batch size, per-item paths].
        ref (Tensor): Reference object IDs, int Tensor in shape of [batch size, per-item paths].
        rl2 (Tensor): reference-hist_item relation IDs, int Tensor in shape of [batch size, per-item paths].
        hist_item (Tensor): Historical item IDs, int Tensor in shape of [batch size, per-item paths].

    Outputs:
        probs (Tensor): Prediction probabilities, float Tensor in shape of [batch size, ].
        label (Tensor): Ground-truth labels, float Tensor in shape of [batch size, ]

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, network):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, item, rl1, ref, rl2, hist_item, label):
        """Forward."""
        outputs = self.network(item, rl1, ref, rl2, hist_item)
        scores = outputs[0]
        probs = self.sigmoid(scores)

        return probs, label
