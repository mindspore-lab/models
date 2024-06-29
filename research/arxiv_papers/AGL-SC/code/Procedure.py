import world
import numpy as np
import mindspore.ops as ops
import mindspore

import utils

from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


def utils2(x, y, p=2):
    wasserstein_distance = ops.abs(
        (
            ops.sort(x.transpose(0, 1), axis=1)[0]
            - ops.sort(y.transpose(0, 1), axis=1)[0]
        )
    )
    wasserstein_distance = ops.pow(
        ops.sum(ops.pow(wasserstein_distance, p), dim=1), 1.0 / p
    )
    wasserstein_distance = ops.pow(ops.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, W, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = mindspore.Tensor(S[:, 0]).astype(mindspore.int64)
    posItems = mindspore.Tensor(S[:, 1]).astype(mindspore.int64)
    negItems = mindspore.Tensor(S[:, 2]).astype(mindspore.int64)
    # print(users)
    users = users
    posItems = posItems
    negItems = negItems
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config["bpr_batch_size"] + 1
    aver_loss = 0.0
    for batch_users, batch_pos, batch_neg in tqdm(
        utils.minibatch(
            users, posItems, negItems, batch_size=world.config["bpr_batch_size"]
        )
    ):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += (
            0.5
            * utils2(
                W,
                ops.mm(
                    Recmodel.embedding_user.weight, Recmodel.embedding_item.weight.t()
                ),
            )
            + cri
        )
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {
        "recall": np.array(recall),
        "precision": np.array(pre),
        "ndcg": np.array(ndcg),
    }


def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    u_batch_size = len(list(testDict.keys()))
    Recmodel: model.PureMF
    # eval mode with no dropout
    Recmodel = Recmodel.set_train(False)
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {
        "ndcg": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "precision": np.zeros(len(world.topks)),
    }

    users = list(testDict.keys())
    users_list = []
    rating_list = []
    groundTrue_list = []
    # auc_record = []
    # ratings = []
    total_batch = len(users) // u_batch_size + 1
    for batch_users in utils.minibatch(users, batch_size=u_batch_size):
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        batch_users_gpu = mindspore.Tensor(batch_users).astype(mindspore.int64)
        batch_users_gpu = batch_users_gpu

        rating = Recmodel.getUsersRating(batch_users_gpu)
        # rating = ratings
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = ops.topk(rating, k=max_K)
        rating = rating.asnumpy()
        # aucs = [
        #         utils.AUC(rating[i],
        #                   dataset,
        #                   test_data) for i, test_data in enumerate(groundTrue)
        #     ]
        # auc_record.extend(aucs)
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K)
        groundTrue_list.append(groundTrue)
    # assert total_batch == len(users_list)
    X = zip(rating_list, groundTrue_list)
    if multicore == 1:
        pre_results = pool.map(test_one_batch, X)
    else:
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
    scale = float(u_batch_size / len(users))
    for result in pre_results:
        results["recall"] += result["recall"]
        results["precision"] += result["precision"]
        results["ndcg"] += result["ndcg"]
    results["recall"] /= float(len(users))
    results["precision"] /= float(len(users))
    results["ndcg"] /= float(len(users))
    # results['auc'] = np.mean(auc_record)
    if world.tensorboard:
        w.add_scalars(
            f"Test/Recall@{world.topks}",
            {
                str(world.topks[i]): results["recall"][i]
                for i in range(len(world.topks))
            },
            epoch,
        )
        w.add_scalars(
            f"Test/Precision@{world.topks}",
            {
                str(world.topks[i]): results["precision"][i]
                for i in range(len(world.topks))
            },
            epoch,
        )
        w.add_scalars(
            f"Test/NDCG@{world.topks}",
            {str(world.topks[i]): results["ndcg"][i] for i in range(len(world.topks))},
            epoch,
        )
    if multicore == 1:
        pool.close()
    # print(results)
    return results
