import time
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class MultiVAE(nn.Cell):

    def __init__(self, config, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
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

    def construct(self, input):
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

        return z, kl_loss + ce_loss

    def encode(self, input):
        L2 = ops.L2Normalize()
        h = L2(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = ops.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = ops.exp(0.5 * logvar)
            eps = ops.randn_like(std)
            return eps.mul(std).add_(mu)
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
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def load_data(train_file, test_file):
    import scipy.sparse as sp

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


def create_RMN(
    train_data, uLen, vLen
):  
    from tqdm import tqdm

    print("构建R矩阵")
    data = []
    W = ops.zeros((uLen, vLen))  
    for w in tqdm(train_data):
        u, v = w
        data.append((u, v))
        W[u][v] = 1  

    return W


###############################################################################
# Training code
###############################################################################


def train(model, optimizer, W):
    # Turn on training mode
    model.train()
    optimizer.zero_grad()
    recon_batch, loss = model(W)
    loss.backward()
    optimizer.step()

    return recon_batch, loss


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.0


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.0
    return dcg_at_k(r, k) / idcg


def eval_ndcg(model, test_matrix, predicted_matrix, k=20):

    predicted_matrix = predicted_matrix.asnumpy()
    test_matrix = test_matrix.asnumpy()
    ndcgs = []
    for user in range(predicted_matrix.shape[0]):
        test_ratings = test_matrix[user]
        predicted_ratings = predicted_matrix[user]
        top_k_predicted = np.argsort(predicted_ratings)[-k:]
        top_k_ratings = test_ratings[top_k_predicted]
        ndcg = ndcg_at_k(top_k_ratings, k)
        ndcgs.append(ndcg)
    return np.mean(ndcgs)

def multivae(config, W):

    np.random.seed(config["seed"])
    mindspore.dataset.config.set_seed(config["seed"])

    train_file = r"../data/" + config["dataset"] + "/train.txt"  
    test_file = r"../data/" + config["dataset"] + "/test.txt"
    print("train_file:", train_file)
    print("test_file:", test_file)
    train_data, test_data, uLen, vLen = load_data(train_file, test_file)
    config["ulen"] = uLen
    config["vlen"] = vLen
    config["n_items"] = vLen
    p_dims = [200, 600, vLen]
    config["p_dims"] = p_dims

    print("创建初始矩阵")
    W2 = create_RMN(test_data, uLen, vLen)

    model = MultiVAE(config)
    optimizer = nn.Adam(
        model.parameters(), learning_rate=config["lr"], weight_decay=config["wd"]
    )

    best = 0.0
    bestR = []
    bestE = 0
    converged_epochs = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(config["epochs"]):
            start_time = time.time()
            R, loss = train(model, optimizer, W)
            val = eval_ndcg(model, W2, R, k=20)
            print("=" * 89)
            print(
                "| Epoch {:2d}|loss {:4.5f} | n20 {:4.5f}".format(
                    epoch, loss, val
                )
            )
            if val > best:
                best = val
                bestE = epoch
                bestR = R
                converged_epochs = 0
            if converged_epochs >= 10 and epoch > 100:
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

    return model, optimizer, bestR
