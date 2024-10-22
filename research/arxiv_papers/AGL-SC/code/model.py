import world

from dataloader import BasicDataset

import mindspore
from mindspore import ops, nn
from mindspore.common.initializer import Normal, initializer


class BasicModel(nn.Cell):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config["latent_dim_rec"]
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = nn.Embedding(
            vocab_size=self.num_users, embedding_size=self.latent_dim
        )
        self.embedding_item = nn.Embedding(
            vocab_size=self.num_items, embedding_size=self.latent_dim
        )
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.astype(mindspore.int64)
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = ops.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.astype(mindspore.int64))
        pos_emb = self.embedding_item(pos.astype(mindspore.int64))
        neg_emb = self.embedding_item(neg.astype(mindspore.int64))
        pos_scores = ops.sum(users_emb * pos_emb, dim=1)
        neg_scores = ops.sum(users_emb * neg_emb, dim=1)
        loss = ops.mean(ops.softplus(neg_scores - pos_scores))
        reg_loss = (
            (1 / 2)
            * (
                users_emb.norm(2).pow(2)
                + pos_emb.norm(2).pow(2)
                + neg_emb.norm(2).pow(2)
            )
            / float(len(users))
        )
        return loss, reg_loss

    def construct(self, users, items):
        users = users.astype(mindspore.int64)
        items = items.astype(mindspore.int64)
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = ops.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


def utils2(x, y, p=2):
    wasserstein_distance = torch.abs(
        (
            torch.sort(x.transpose(0, 1), dim=1)[0]
            - torch.sort(y.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.pow(
        torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p
    )
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance


class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset, q_dims=None, dropout=0.5):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

        # vae
        p_dims = [200, 600, self.dataset.m_items]
        self.p_dims = p_dims
        if q_dims:
            assert (
                q_dims[0] == p_dims[-1]
            ), "In and Out dimensions must equal to each other"
            assert (
                q_dims[-1] == p_dims[0]
            ), "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.CellList(
            [
                nn.Dense(d_in, d_out)
                for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])
            ]
        )
        self.p_layers = nn.CellList(
            [
                nn.Dense(d_in, d_out)
                for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])
            ]
        )

        self.update = 0
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.mlplist = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(128, 128 * 2), nn.ReLU(), nn.Dense(2 * 128, 128)
                )
            ]
        )

    def init_weights(self):
        for layer in self.q_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(
                layer.weight.data, nonlinearity="tanh"
            )

            # Normal Initialization for Biases
            layer.bias.data.set_data(
                initializer("normal", layer.bias.data.shape, layer.bias.data.dtype)
            )

        for layer in self.p_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(
                layer.weight.data, nonlinearity="tanh"
            )

            # Normal Initialization for Biases
            layer.bias.data.set_data(
                initializer("normal", layer.bias.data.shape, layer.bias.data.dtype)
            )

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lightGCN_n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.A_split = self.config["A_split"]
        self.embedding_user = nn.Embedding(
            vocab_size=self.num_users, embedding_size=self.latent_dim
        )
        self.embedding_item = nn.Embedding(
            vocab_size=self.num_items, embedding_size=self.latent_dim
        )
        if self.config["pretrain"] == 0:

            self.embedding_user.embedding_table.set_data(
                initializer(
                    "normal",
                    self.embedding_user.embedding_table.shape,
                    self.embedding_user.embedding_table.dtype,
                )
            )
            self.embedding_item.embedding_table.set_data(
                initializer(
                    "normal",
                    self.embedding_item.embedding_table.shape,
                    self.embedding_item.embedding_table.dtype,
                )
            )

            # world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.embedding_table.data.copy(
                mindspore.Tensor.from_numpy(self.config["user_emb"])
            )
            self.embedding_item.embedding_table.data.copy(
                mindspore.Tensor.from_numpy(self.config["item_emb"])
            )
            print("use pretarined data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = ops.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = mindspore.Tensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.embedding_table
        items_emb = self.embedding_item.embedding_table
        all_emb = ops.cat([users_emb, items_emb])

        embs = [all_emb]
        # if self.config["dropout"]:
        #     if self.training:
        #         print("droping")
        #         g_droped = self.__dropout(self.keep_prob)
        #     else:
        #         g_droped = self.Graph
        # else:
        g_droped = self.Graph
        from mindspore.common.initializer import Zero

        # print(g_droped.values)
        g_droped = g_droped.to_dense()
        # print(g_droped)

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(g_droped[f].mm(all_emb))
                side_emb = ops.cat(temp_emb, axis=0)
                all_emb = side_emb
            else:
                all_emb = g_droped.mm(all_emb)
            embs.append(all_emb)
        embs = ops.stack(embs, axis=1)
        light_out = ops.mean(embs, axis=1)
        # light_out = light_out.to_dense()
        # light_out = mindspore.COOTensor(light_out.asnumpy()).to_dense()
        users, items = ops.split(light_out, [self.num_users, self.num_items], axis=0)
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.astype(mindspore.int64)]
        items_emb = all_items
        rating = self.f(ops.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
