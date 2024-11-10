import numpy as np
import os
import math

import mindspore
from mindspore.common.initializer import initializer, Normal
from mindspore import ops, nn
from sklearn import preprocessing
import numpy as np
import evaluating_indicator

from tqdm import tqdm
import os
import time


def read_dataset(filename):
    orgin_data = []
    with open(filename, encoding="UTF-8") as fin:
        line = fin.readline()
        while line:
            user, item, rating = line.strip().split("\t")
            orgin_data.append((int(user) - 1, int(item) - 1, float(rating)))
            line = fin.readline()

    user, item = set(), set()
    for u, v, r in orgin_data:
        user.add(u)
        item.add(v)
    user_list = list(user)
    item_list = list(item)
    uLen = max(user_list) + 1
    vLen = max(item_list) + 1
    return orgin_data, user_list, item_list, uLen, vLen


def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, "r") as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip("\n").split(" ")
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip("\n").split(" ")
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1
    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])

    return train_data, test_data, n_user, m_item


def read_data(train_file, test_file):
    train_data = []
    test_data = []
    with open(train_file, encoding="UTF-8") as fin:
        line = fin.readline()
        while line:
            user, item, rating = line.strip().split("\t")
            train_data.append((int(user) - 1, int(item) - 1, float(rating)))
            line = fin.readline()
    user, item, _ = zip(*train_data)
    num_users = max(user) + 1
    num_items = max(item) + 1
    with open(test_file, encoding="UTF-8") as fin:
        line = fin.readline()
        while line:
            user, item, rating = line.strip().split("\t")
            test_data.append((int(user) - 1, int(item) - 1, float(rating)))
            line = fin.readline()
    user, item, _ = zip(*test_data)
    num_users = max(max(user) + 1, num_users)
    num_items = max(max(item) + 1, num_items)

    return train_data, num_users, num_items


def create_RMN(train_data, uLen, vLen):
    data = []
    W = mindspore.numpy.zeros((uLen, vLen))
    indices = mindspore.numpy.array(
        [[u, v] for u, v in train_data], dtype=mindspore.int32
    )
    values = mindspore.numpy.ones(len(train_data), dtype=mindspore.float32)

    W = ops.tensor_scatter_update(W, indices, values)

    return W


def sumpow(x, k):
    sum = 0
    for i in range(k + 1):
        sum += math.pow(x, i)
    return sum


def computeResult(net, test_data):
    node_list_u_, node_list_v_ = {}, {}
    test_user, test_item, test_rate = test_data
    i = 0
    for item in net.u.weight:
        node_list_u_[i] = {}
        node_list_u_[i]["embedding_vectors"] = item.asnumpy()
        i += 1

    i = 0
    for item in net.v.weight:
        node_list_v_[i] = {}
        node_list_v_[i]["embedding_vectors"] = item.asnumpy()
        i += 1
    f1, map, mrr, mndcg = evaluating_indicator.top_N(
        net, test_user, test_item, test_rate, node_list_u_, node_list_v_, top_n=10
    )
    print("f1:", f1, "map:", map, "mrr:", mrr, "mndcg:", mndcg)
    return mndcg


def utils2(x, y, p=2):
    wasserstein_distance = ops.abs(
        (ops.sort(x.swapaxes(0, 1), axis=1)[0] - ops.sort(y.swapaxes(0, 1), axis=1)[0])
    )
    wasserstein_distance = ops.pow(
        ops.sum(ops.pow(wasserstein_distance, p), dim=1), 1.0 / p
    )
    wasserstein_distance = ops.pow(ops.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance


def init_vectors(rank, uLen, vLen):
    print("初始化向量")

    u_vectors = np.random.random([uLen, rank])
    mindspore.common.initializer.HeNormal(u_vectors, nonlinearity="leaky_relu")
    v_vectors = np.random.random([vLen, rank])
    mindspore.common.initializer.HeNormal(v_vectors, nonlinearity="leaky_relu")

    return u_vectors, v_vectors


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):

    neg_candidates = np.arange(item_num)
    if sampling_sift_pos:
        neg_items = []
        for u in pos_train_data[0]:
            probs = np.ones(item_num)
            probs[interacted_items[u]] = 0
            probs /= np.sum(probs)

            u_neg_items = np.random.choice(
                neg_candidates, size=neg_ratio, p=probs, replace=True
            ).reshape(1, -1)

            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(
            neg_candidates, (len(pos_train_data[0]), neg_ratio), replace=True
        )

    neg_items = mindspore.Tensor.from_numpy(neg_items)

    return pos_train_data[0], pos_train_data[1], neg_items


class GRU_Cell(nn.Cell):

    def __init__(self, in_dim, hidden_dim):
        super(GRU_Cell, self).__init__()
        self.rx_linear = nn.Dense(in_dim, hidden_dim)
        self.rh_linear = nn.Dense(hidden_dim, hidden_dim)
        self.zx_linear = nn.Dense(in_dim, hidden_dim)
        self.zh_linear = nn.Dense(hidden_dim, hidden_dim)
        self.hx_linear = nn.Dense(in_dim, hidden_dim)
        self.hh_linear = nn.Dense(hidden_dim, hidden_dim)

    def construct(self, x, h_1):
        r = ops.sigmoid(self.rx_linear(x) + self.rh_linear(h_1))
        z = ops.sigmoid(self.zx_linear(x) + self.zh_linear(h_1))
        h_ = ops.tanh(self.hx_linear(x) + self.hh_linear(r * h_1))
        h = z * h_1 + (1 - z) * h_
        return h


class Net(nn.Cell):
    def __init__(self, config, u_vectors, v_vectors, q_dims=None, dropout=0.5):
        super(Net, self).__init__()
        u_len = config["ulen"]
        v_len = config["vlen"]
        rank = config["rank"]
        self.u = nn.Embedding(u_len, rank)
        self.v = nn.Embedding(v_len, rank)

        # vae
        p_dims = config["p_dims"]
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

        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1]]
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

        self.mlplist = nn.CellList(
            [
                nn.SequentialCell(
                    nn.Dense(rank, rank * 2), nn.ReLU(), nn.Dense(2 * rank, rank)
                )
            ]
        )
        self.vaelist = nn.CellList(
            [nn.SequentialCell(nn.Dense(v_len, 600), nn.Dense(600, 400))]
        )

        self.GRU = nn.GRU(input_size=1, hidden_size=200, batch_first=True)

        self.rnn_cell = GRU_Cell(in_dim=1, hidden_dim=400)
        self.init_weights()

    def gru(self, x, h=None):
        if h is None:
            h = ops.zeros(x.shape[0], self.hidden_dim)
        outs = []
        for t in range(x.shape[1]):
            seq_x = x[:, t, :]
            h = self.rnn_cell(seq_x, h)
            outs.append(ops.unsqueeze(h, 1))

        outs = ops.cat(outs, axis=1)
        return outs, h

    def nfm(self):
        interaction = ops.mm(self.u.embedding_table, self.v.embedding_table.t())

        square_of_sum = ops.pow(interaction, 2)

        self.fc_layers = nn.CellList()
        hidden_dims = [600, 400, 200]
        for idx, (in_size, out_size) in enumerate(
            zip([vLen] + hidden_dims[:-1], hidden_dims)
        ):
            self.fc_layers.append(nn.Dense(in_size, out_size))
            self.fc_layers.append(nn.ReLU())
        inter_term = 0.5 * (square_of_sum)
        for layer in self.fc_layers:
            inter_term = layer(inter_term)
        output = inter_term

        return output

    def encode(self, input):
        L2 = ops.L2Normalize()
        h = L2(input)
        h = self.drop(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = ops.tanh(h)
            else:
                W = self.nfm()
                W = W.unsqueeze(0)
                h = h.unsqueeze(-1)
                output, h = self.GRU(h, W)
                h1 = h.squeeze(0)
                h2 = output.sum(axis=1)
                mu = h1
                logvar = h2

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = ops.exp(0.5 * logvar)
            eps = ops.rand_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = ops.tanh(h)
        return h

    def init_weights(self):
        mindspore.common.initializer.HeNormal(
            self.u.embedding_table, nonlinearity="leaky_relu'"
        )
        mindspore.common.initializer.HeNormal(
            self.v.embedding_table, nonlinearity="leaky_relu'"
        )
        for layer in self.q_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(
                layer.weight.data, nonlinearity="leaky_relu'"
            )

            # Normal Initialization for Biases
            mindspore.common.initializer.HeNormal(
                layer.bias.data, nonlinearity="leaky_relu'"
            )

        for layer in self.p_layers:
            # Kaiming Initialization for weights
            mindspore.common.initializer.HeNormal(
                layer.weight.data, nonlinearity="leaky_relu'"
            )

            # Normal Initialization for Biases
            mindspore.common.initializer.HeNormal(
                layer.bias.data, nonlinearity="leaky_relu'"
            )

    def vae(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        kl_loss = (
            -0.5
            * ops.mean(ops.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        ce_loss = -(ops.log_softmax(z, 1) * input).sum(1).mean()

        return kl_loss + ce_loss, z

    def mlp(self):
        combined_embedding = ops.cat(
            (self.u.embedding_table, self.v.embedding_table), axis=0
        )
        for layer in self.mlplist:
            combined_embedding = layer(combined_embedding)
        user_mask, item_mask = ops.split(combined_embedding, [uLen, vLen], axis=0)

        return ops.mm(user_mask, item_mask.t())

    def construct(self, W):
        RR = ops.mm(self.u.embedding_table, self.v.embedding_table.t())
        logp_R = ops.log_softmax(RR, axis=-1)  
        p_R = ops.softmax(W, axis=-1)  
        kl_sum_R = ops.KLDivLoss(reduction="sum")(logp_R, p_R)
        # C = utils2(W, RR)
        loss = kl_sum_R
        return loss, RR


def train_mlp(config, model, optimizer, W, W2, k):
    best = 0.0
    best2 = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0
    try:
        should_stop = False
        for epoch in range(config["epochs"]):
            start_time = time.time()
            model.set_train(True)
            R, loss = model.construct1(W)

            loss.backward()
            optimizer.step()
            model.set_train(False)
            import multivae1

            ndcg = multivae1.eval_ndcg(model, W2, R, k=k)
            print(
                "|Epoch",
                epoch,
                "|NDCG:",
                ndcg,
                "|Loss",
                loss.item(),
            )
            converged_epochs += 1
            if ndcg > best:
                best = ndcg
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 10 and epoch > 50:
                print(
                    "模型收敛，停止训练。最优ndcg值为：",
                    best,
                    "最优epoch为：\n",
                    bestE,
                    "最优R为：\n",
                    bestR,
                )
                print("保存模型参数")
                break
            if epoch == config["epochs"] - 1:
                print("模型收敛，停止训练。最优ndcg值为：", best, "最优R为：\n", bestR)
            if should_stop:
                break
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    print("=" * 50)
    train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
    print("训练时间:", train_time)
    print(
        "| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}".format(bestE, config["topk"], best)
    )
    print("=" * 50)

    return bestR


def train_gnn(config, model, optimizer, W, W2):
    best = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0

    def forward_fn(W1):
        loss, RR = model(W1)
        return loss, RR

    try:
        should_stop = False
        for epoch in range(config["epochs"]):
            start_time = time.time()
            model.set_train()
            grad_fn = mindspore.value_and_grad(
                forward_fn, None, optimizer.parameters, has_aux=True
            )
            (loss, R), grads = grad_fn(W)
            optimizer(grads)
            model.set_train(False)
            from multivae1 import eval_ndcg

            val = eval_ndcg(model, W2, R, k=20)
            print("| 训练轮次:{0}  |NDCG@{1}:{2}".format(epoch, config["topk"], val))
            converged_epochs += 1
            if val > best:
                best = val
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 20 and epoch > 150:
                print(
                    "模型收敛，停止训练。最优ndcg值为：",
                    best,
                    "最优epoch为：\n",
                    bestE,
                    "最优R为：\n",
                    bestR,
                )
                print("保存模型参数")
                break
            if epoch == config["epochs"] - 1:
                print("模型收敛，停止训练。最优ndcg值为：", best, "最优R为：\n", bestR)
            if should_stop:
                break
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    print("=" * 50)
    train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
    print("训练时间:", train_time)
    print(
        "| 训练结束 | 最优轮:{0}|最优 NDCG@{1}:{2}".format(bestE, config["topk"], best)
    )
    print("=" * 50)

    return bestR


config = {
    "dataset": "Movielens1M",
    "topk": 40,
    "lr": 1e-3,
    "wd": 0.0,
    "rank": 128,
    "batch_size": 512,
    "testbatch": 100,
    "epochs": 1000,
    "total_anneal_steps": 200000,
    "anneal_cap": 0.2,
    "seed": 2024,
}

if __name__ == "__main__":
    from evalu import run
    import os

    mindspore.set_context(device_target="GPU", device_id=1)
    print("Random seed: {}".format(config["seed"]))
    np.random.seed(config["seed"])
    mindspore.dataset.config.set_seed(config["seed"])

    eval = run(config["dataset"])
    rank = 128
    epochs = config["epochs"]
    lr = config["lr"]
    iters = 1
    top_n = config["topk"]
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    test_batch_size = config["testbatch"]

    print(
        "dataset",
        dataset,
        "rank:",
        rank,
        "epochs:",
        epochs,
        "lr:",
        lr,
        "topK@",
        top_n,
        "iters:",
        iters,
    )
    path = os.path.dirname(os.path.dirname(__file__))
    train_file = r"../data/" + dataset + "/train.txt"
    test_file = r"../data/" + dataset + "/test.txt"
    print("train_file:", train_file)
    print("test_file:", test_file)
    train_data, test_data, uLen, vLen = load_data(train_file, test_file)
    config["ulen"] = uLen
    config["vlen"] = vLen
    config["n_items"] = vLen
    p_dims = [200, 600, vLen]
    config["p_dims"] = p_dims
    print("用户项目数：", uLen, vLen)
    print("创建初始W,W2矩阵")
    W = create_RMN(train_data, uLen, vLen)
    W2 = create_RMN(test_data, uLen, vLen)

    m = None
    n = None

    replaced_indices = []

    # kaiming_init
    u_vector, v_vector = init_vectors(rank, uLen, vLen)
    u_vectors = mindspore.Tensor.from_numpy(u_vector).astype(mindspore.float32)
    v_vectors = mindspore.Tensor.from_numpy(v_vector).astype(mindspore.float32)

    model = Net(config, u_vectors, v_vectors)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=config["lr"])

    R_fake_path = r"R_" + dataset + ".ckpt"
    R_fake_np = r"R_" + dataset + ".npy"
    z_path = r"R_vae_" + dataset + ".ckpt"
    z_np = r"R_vae_" + dataset + ".npy"
    R_fake = W
    z = W

    if os.path.exists(R_fake_path) and os.path.exists(R_fake_np):
        param_dict = mindspore.load_checkpoint(R_fake_path)

        model_params = {
            name: param for name, param in param_dict.items() if "model." in name
        }
        mindspore.load_param_into_net(model, model_params)
        optimizer_params = {
            name: param for name, param in param_dict.items() if "optimizer." in name
        }
        mindspore.load_param_into_net(optimizer, optimizer_params)

        np_array = np.load(R_fake_np)
        R_fake = mindspore.Tensor(np_array)

        print("载入训练矩阵R.")

    else:

        print("重新训练矩阵R.")
        R_fake = train_gnn(config, model, optimizer, R_fake, W2)
        model_params = {
            f"model.{name}": param for name, param in model.parameters_and_names()
        }
        optimizer_params = {
            f"optimizer.{name}": param
            for name, param in optimizer.parameters_and_names()
        }
        save_obj = {**model_params, **optimizer_params}
        mindspore.save_checkpoint(save_obj, R_fake_path)
        np.save(R_fake_np, R_fake.asnumpy())
