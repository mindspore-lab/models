from pip import main

import numpy as np

from mindspore import ops

import math

import numpy as np


def read_data(filename=None):
    users, items, rates = set(), set(), {}
    with open(filename, "r", encoding="UTF-8") as fin:
        line = fin.readline()
        while line:
            user, item, rate = line.strip().split()
            user = int(user) - 1
            item = int(item) - 1
            if rates.get(user) is None:
                rates[user] = {}
            rates[user][item] = float(rate)
            users.add(user)
            items.add(item)
            line = fin.readline()
    return users, items, rates


def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0


def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0


def precision_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits / (1.0 * len(ranked_list))
    rec = hits / (1.0 * len(ground_list))
    return pre, rec


def top_N1(net, test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
    recommend_dict = {}
    list_u = list(test_u)
    list_v = list(test_v)

    uv = ops.matmul(net.u.weight[list_u], net.v.weight[list_v].t())

    def get_mapping(input_num, mapping_dict):
        if input_num in mapping_dict:
            return mapping_dict[input_num]
        else:
            return -1

    mapping_dict_u = {key: value for value, key in enumerate(set(test_u))}
    mapping_dict_i = {key: value for value, key in enumerate(set(test_v))}

    for u in test_u:
        u1 = get_mapping(u, mapping_dict_u)
        recommend_dict[u] = {}
        for v in test_v:
            v1 = get_mapping(v, mapping_dict_i)
            pre = uv[u1][v1]

            recommend_dict[u][v] = float(pre)
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u in test_u:

        tmp_r = sorted(recommend_dict[u].items(), key=lambda x: x[1], reverse=True)[
            0 : min(len(recommend_dict[u]), top_n)
        ]
        tmp_t = sorted(test_rate[u].items(), key=lambda x: x[1], reverse=True)[
            0 : min(len(test_rate[u]), top_n)
        ]

        tmp_r_list = []
        tmp_t_list = []

        for item, rate in tmp_r:
            tmp_r_list.append(item)

        for item, rate in tmp_t:
            tmp_t_list.append(item)
        pre, rec = precision_and_recall(tmp_r_list, tmp_t_list)
        ap = AP(tmp_r_list, tmp_t_list)
        rr = RR(tmp_r_list, tmp_t_list)
        ndcg = nDCG(tmp_r_list, tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
        precison = sum(precision_list) / len(precision_list)

    recall = sum(recall_list) / len(recall_list)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1, map, mrr, mndcg


def top_N(test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
    recommend_dict = {}
    for u in test_u:
        recommend_dict[u] = {}
        for v in test_v:
            if node_list_u.get(u) is None:
                pre = 0
            else:
                U = np.array(node_list_u[u]["embedding_vectors"])
                if node_list_v.get(v) is None:
                    pre = 0
                else:
                    V = np.array(node_list_v[v]["embedding_vectors"])
                    pre = U.dot(V.T)

            recommend_dict[u][v] = float(pre)
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []

    for u in test_u:

        tmp_r = sorted(recommend_dict[u].items(), key=lambda x: x[1], reverse=True)[
            0 : min(len(recommend_dict[u]), top_n)
        ]
        tmp_t = sorted(test_rate[u].items(), key=lambda x: x[1], reverse=True)[
            0 : min(len(test_rate[u]), top_n)
        ]

        tmp_r_list = []
        tmp_t_list = []

        for item, rate in tmp_r:
            tmp_r_list.append(item)

        for item, rate in tmp_t:
            tmp_t_list.append(item)
        pre, rec = precision_and_recall(tmp_r_list, tmp_t_list)
        ap = AP(tmp_r_list, tmp_t_list)
        rr = RR(tmp_r_list, tmp_t_list)
        ndcg = nDCG(tmp_r_list, tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
        precison = sum(precision_list) / len(precision_list)

    recall = sum(recall_list) / len(recall_list)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1, map, mrr, mndcg
