import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from tqdm import tqdm


def create_RMN(train_data, uLen, vLen):
    data = []
    W = torch.zeros((uLen, vLen))
    for w in tqdm(train_data):
        u, v = w
        data.append((u, v))
        W[u][v] = 1

    return W


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


def computeResult(uweight, vweight, test_data, W):
    import evaluating_indicator

    test_rate = [0 for _ in range(uLen)]
    for u in range(uLen):
        if torch.sum(W[u, :]) > 0:
            test_rate[u] = 1
    node_list_u_, node_list_v_ = {}, {}
    test_user, test_item = zip(*test_data)
    i = 0
    for item in uweight:
        node_list_u_[i] = {}
        node_list_u_[i]["embedding_vectors"] = item.cpu().detach().numpy()
        i += 1

    # 对于v 需要在这里映射一下
    i = 0
    for item in vweight:
        node_list_v_[i] = {}
        node_list_v_[i]["embedding_vectors"] = item.cpu().detach().numpy()
        i += 1
    f1, map, mrr, mndcg = evaluating_indicator.top_N(
        test_user, test_item, test_rate, node_list_u_, node_list_v_, top_n=10
    )
    print("f1:", f1, "map:", map, "mrr:", mrr, "mndcg:", mndcg)
    return mndcg


world.config["wd"] = 0.0
world.config["total_anneal_steps"] = 200000
world.config["anneal_cap"] = 0.2
world.config["topk"] = 20
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(
            torch.load(weight_file, map_location=torch.device("cpu"))
        )
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    best = 0.0
    bestuser = []
    bestitem = []
    bestE = 0
    converged_epochs = 0

    path = world.config["dataset"]

    train_file = r"../data/" + path + "/train.txt"
    test_file = r"../data/" + path + "/test.txt"
    train_data, test_data, uLen, vLen = load_data(train_file, test_file)
    W = create_RMN(train_data, uLen, vLen).to(world.config["device"])
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 1 == 0:
            cprint("[TEST]")
            result = Procedure.Test(
                dataset, Recmodel, epoch, w, world.config["multicore"]
            )
        val = result["ndcg"]
        converged_epochs += 1
        print(result)
        if val > best:
            best = val
            bestE = epoch
            bestuser = Recmodel.embedding_user.weight
            bestitem = Recmodel.embedding_item.weight
            converged_epochs = 0
        if converged_epochs >= 10 and epoch > 1500:
            print("模型收敛，停止训练。最优ndcg值为：", best, "最优epoch为：\n", bestE)
            break
        output_information = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, W, neg_k=Neg_k, w=w
        )
        print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}")

    cprint("保存参数：")
finally:
    cprint("保存参数：")
    if world.tensorboard:
        w.close()
