import numpy as np
import mindspore
from mindspore import Tensor
from mindspore import numpy as mnp
import numpy as np



# 均方误差（Mean Square Error）
def mse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MSE: 预测数据与真实数据大小不一致'
    return np.mean(sum(pow(loc_pred - loc_true, 2)))


# 平均绝对误差（Mean Absolute Error）
def mae(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAE: 预测数据与真实数据大小不一致'
    return np.mean(sum(loc_pred - loc_true))


# 均方根误差（Root Mean Square Error）
def rmse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'RMSE: 预测数据与真实数据大小不一致'
    return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))


# 平均绝对百分比误差（Mean Absolute Percentage Error）
def mape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAPE: 预测数据与真实数据大小不一致'
    assert 0 not in loc_true, "MAPE: 真实数据有0，该公式不适用"
    return np.mean(abs(loc_pred - loc_true) / loc_true)


# 平均绝对和相对误差（Mean Absolute Relative Error）
def mare(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "MARE：预测数据与真实数据大小不一致"
    assert np.sum(loc_true) != 0, "MARE：真实位置全为0，该公式不适用"
    return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)


# 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
def smape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'SMAPE: 预测数据与真实数据大小不一致'
    assert 0 in (loc_pred + loc_true), "SMAPE: 预测数据与真实数据有0，该公式不适用"
    return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) +
                                                        np.abs(loc_true)))


# 对比真实位置与预测位置获得预测准确率
def acc(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "accuracy: 预测数据与真实数据大小不一致"
    loc_diff = loc_pred - loc_true
    loc_diff[loc_diff != 0] = 1
    return loc_diff, np.mean(loc_diff == 0)


def top_k(loc_pred, loc_true, k):
    """
    count the hit numbers of loc_true in topK of loc_pred, used to calculate Precision, Recall and F1-score,
    calculate the reciprocal rank, used to calcualte MRR,
    calculate the sum of DCG@K of the batch, used to calculate NDCG

    Args:
        loc_pred: (batch_size * output_dim)
        loc_true: (batch_size * 1)
        topk:

    Returns:
        tuple: tuple contains:
            hit (int): the hit numbers \n
            rank (float): the sum of the reciprocal rank of input batch \n
            dcg (float): dcg
    """
    assert k > 0, "top-k ACC评估方法：k值应不小于1"
    loc_pred=loc_pred.asnumpy()
    loc_true=loc_true.asnumpy()
    # loc_pred = Tensor(loc_pred,mindspore.float16)
    # print(loc_pred.shape)
    correct_num = 0.0  # 预测正确个数
    total_num = 0.0
    # val, index = mindspore.ops.topk(loc_pred,k,1)
    pred = np.argmax(loc_pred,axis=1)
    correct = np.equal(pred, loc_true).reshape((-1, ))
    correct_num += correct.sum()
    total_num += correct.shape[0]
    # print("[{}]".format(correct_num))
    return correct_num,total_num

sort=mindspore.ops.Sort(descending=True)
def Precision_m(preds, labels, topk):
    precision = []
    preds=preds.asnumpy().flatten()
    labels=labels.asnumpy().flatten()
    for i in range(preds.shape[0]):
        label = labels[i]
        pred = preds[i]
        accident_grids = label > 0
        accident_grids = accident_grids.astype(int) 
        idx=np.argsort(-preds)
        sorted=preds[idx]
        threshold = sorted[topk - 1]
        pred_grids = pred >= threshold
        pred_grids = pred_grids.astype(int)
        matched = (pred_grids + accident_grids) == 2
        matched = matched.astype(float)
        precision.append(matched.sum() / topk)
    return sum(precision) / len(precision)

def Recall_m(preds, labels, topk):
    recall = []
    preds=preds.asnumpy().flatten()
    labels=labels.asnumpy().flatten()
    for i in range(preds.shape[0]):
        label = labels[i]
        pred = preds[i]
        accident_grids = label > 0
        accident_grids = accident_grids.astype(int) 
        idx=np.argsort(-preds)
        sorted=preds[idx]
        threshold = sorted[topk - 1]
        pred_grids = pred >= threshold
        pred_grids = pred_grids.astype(int)
        matched = (pred_grids + accident_grids) == 2
        matched = matched.astype(float)
        if np.sum(accident_grids) != 0:
            recall.append(matched.sum() / accident_grids.sum())
    return sum(recall) / len(recall)

def F1_Score_m(preds, labels, topk):
    precision = Precision_m(preds, labels, topk)
    recall = Recall_m(preds, labels, topk)
    return 2 * precision * recall / (precision + recall)



def MAP_m(preds, labels, topk):
    ap = []
    for i in range(preds.shape[0]):
        label = labels[i].flatten()
        pred = preds[i].flatten()
        accident_grids = label > 0
        sorted, rank = sort(pred)
        rank = rank[:topk]
        if topk != 0:
            threshold = sorted[topk - 1]
        else:
            threshold = 0
        label = label != 0
        pred = pred >= threshold
        matched = pred == label
        match_num = 0
        precision_sum = 0
        for i in range(rank.shape[0]):
            if matched[rank[i]]:
                match_num += 1
                precision_sum += match_num / (i + 1)
        if rank.shape[0] != 0:
            ap.append(precision_sum / rank.shape[0])
    return sum(ap) / len(ap)


def PCC_m(preds, labels, topk):
    pcc = []
    for i in range(preds.shape[0]):
        label = labels[i].flatten()
        pred = preds[i].flatten()
        sorted, rank = sort(pred)
        pred = sorted[:topk]
        rank = rank[:topk]
        sorted_label = mnp.zeros(topk)
        for i in range(topk):
            sorted_label[i] = label[rank[i]]
        label = sorted_label
        label_average = mnp.sum(label) / (label.shape[0])
        pred_average = mnp.sum(pred) / (pred.shape[0])
        if mnp.sqrt(mnp.sum((label - label_average) * (label - label_average))) * mnp.sqrt(
                mnp.sum((pred - pred_average) * (pred - pred_average))) != 0:
            pcc.append((mnp.sum((label - label_average) * (pred - pred_average)) / (
                    mnp.sqrt(mnp.sum((label - label_average) * (label - label_average))) * mnp.sqrt(
                    mnp.sum((pred - pred_average) * (pred - pred_average))))))
    return sum(pcc) / len(pcc)
