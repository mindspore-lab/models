# conda env: rsmt
# -*- coding: utf-8 -*-
# @Time        : 2022/9/7 16:20
# @Author      : gwsun
# @Project     : RSMT
# @File        : myutil.py
# @env         : PyCharm
# @Description :
from utils.rsmt_utils import Evaluator, plot_rest, plot_gst_rsmt
from utils.distance import get_distance
import matplotlib.pyplot as plt
import numpy as np
import torch

evaluator = Evaluator()
def get_length(input):
    """
    :param input: graph node coodinate
    :return: opt length
    """
    len, _, _ = evaluator.gst_rsmt(input)
    return len


def get_length_batch(input):
    batch = input.shape[0]
    len = []
    for i in range(batch):
        len.append(get_length(input[i]))
    return np.array(len)


# 评估所生成的结果的长度
def eval_length(coodinate, edges, degree):
    return evaluator.eval_batch(coodinate, edges, degree)


def eval_single(coodinate, edges, degree):
    return evaluator.eval_func(coodinate.reshape(-1), edges, degree)


def plot_contrast_graph(case, output, add_edge, drop_edges):
    fig = plt.figure(figsize=(15, 4.6))
    plt.subplot(1, 3, 1)
    plt.title("concat graph")
    plot_rest(case, output)
    u, v = add_edge[0], add_edge[1]
    plt.plot([case[u][0], case[u][0]], [case[u][1], case[v][1]], '-', color='m')
    plt.plot([case[u][0], case[v][0]], [case[v][1], case[v][1]], '-', color='m')
    _, sp_list, edges = evaluator.gst_rsmt(case)
    plt.subplot(1, 3, 2)
    plt.title("optimal graph")
    plot_gst_rsmt(case, sp_list, edges)

    plt.subplot(1, 3, 3)
    plt.title("drop_edge graph")
    plot_rest(case, drop_edges)
    fig.savefig('contrast.png')


def plot_merge_tune(arr1, arr2, add_edge, ours_adj, ori_len, lengths, opt_len):

    plt.subplot(1, 3, 1)
    # plt.title("original concat directly")
    _, sp_list, edges = evaluator.gst_rsmt(arr1)
    plot_gst_rsmt(arr1, sp_list, edges)
    _, sp_list, edges = evaluator.gst_rsmt(arr2)
    plot_gst_rsmt(arr2, sp_list, edges)
    u, v = add_edge[0], add_edge[1]
    case = np.concatenate([arr1, arr2], axis=0)
    plt.plot([case[u][0], case[u][0]], [case[u][1], case[v][1]], '-', color='m')
    plt.plot([case[u][0], case[v][0]], [case[v][1], case[v][1]], '-', color='m')
    plt.annotate('Length=' + str(round(ori_len.item(), 6)), (-0.04, -0.04))

    plt.subplot(1, 3, 2)
    # plt.title("our tune on concat graph")
    edges = np.stack(np.where(ours_adj == 1), 0).T.reshape(-1)
    plot_rest(case, edges)
    plt.annotate('Length=' + str(round(lengths.item(), 6)), (-0.04, -0.04))

    plt.subplot(1, 3, 3)
    # plt.title("optimal graph")
    _, sp_list, edges = evaluator.gst_rsmt(case)
    plot_gst_rsmt(case, sp_list, edges)
    plt.annotate('Length=' + str(round(opt_len.item(), 6)), (-0.04, -0.04))


def plot_directed_graph(case, src, dst, edge_color, term_color):
    for u, v in zip(src, dst):
        plt.annotate('', xy=(case[v][0], case[v][1]), xytext=(case[u][0], case[u][1]),
                     arrowprops=dict(color=edge_color, arrowstyle="->"))
    plt.plot([point[0] for point in case], [point[1] for point in case], 's', markerfacecolor='black',
             color=term_color, markersize=4, markeredgewidth=.5)


def eval_len_from_adj(arr, degree, new_adj):  # new_adj会变成对角线全零的
    if not torch.is_tensor(arr):
        arr = torch.from_numpy(arr)
    if not torch.is_tensor(new_adj):
        new_adj = torch.from_numpy(new_adj)
    if len(arr.shape) == 2:
        arr = arr.unsqueeze(0)  # 扩成三维
    if len(new_adj.shape) == 2:
        new_adj[list(range(degree)), list(range(degree))] = 0
        new_adj = new_adj.unsqueeze(0)
    else:
        new_adj[:, list(range(degree)), list(range(degree))] = 0
    batch_size = new_adj.shape[0]
    edge = torch.stack(torch.where(new_adj == 1)).transpose(0, 1).detach().cpu().numpy()
    edge = np.delete(edge, 0, axis=1).reshape(batch_size, -1)
    return eval_length(arr.detach().cpu().numpy(), edge, degree)


def eval_adj_unbanlance(adj, input_batch, valid_degree):
    lengths = []
    batch_size = len(adj)
    max_degree = adj.shape[1]
    adj[:, list(range(max_degree)), list(range(max_degree))] = 0
    for i in range(batch_size):
        edge = torch.stack(torch.where(adj[i] == 1)).transpose(0, 1).detach().cpu().numpy()
        lengths.append(evaluator.eval_func(input_batch[i].cpu().numpy().reshape(-1), edge.reshape(-1), valid_degree[i]))
    return np.array(lengths)

def single_distance(nodes, index, root):
    edges = np.array(index).reshape(-1, 2)  # 2 * (n-1)
    edges = list(map(tuple, edges))
    nodes = list(map(tuple, nodes))
    a = [[nodes[i], nodes[j]] for i, j in edges]
    b = nodes[root]
    leng = get_distance(a, b)[1]
    return leng

def eval_distance(nodes_coo, indexs: list, roots: list):
    """
    nodes_coo : b*n*2
    indexs: 2(n-1)*b
    2-d source：b
    """
    distances = []
    with torch.no_grad():
        edges_all = torch.stack(indexs).transpose(0, 1)  # b*2(n-1)
    for nodes, edges, root in zip(nodes_coo, edges_all, roots):
        # nodes n*2
        # root tuple
        edges = edges.reshape(-1, 2).cpu().numpy()  # 2 * (n-1)
        edges = list(map(tuple, edges))
        # print(edges)
        nodes = list(map(tuple, nodes.cpu().numpy()))
        a = [[nodes[i], nodes[j]]for i, j in edges]
        b = nodes[root]
        leng = get_distance(a, b)[1]
        distances.append(leng)
    return distances


if __name__ == '__main__':
    # arr = np.random.rand(3, 2)
    # length = get_length(arr)
    # print(length)
    # arr = np.array([[[0.1, 0.1], [0.3, 0.2], [0.2, 0.3], [0, 0]]])
    # arr2 = np.array([[0.1, 0.1], [0.3, 0.2], [0.2, 0.3]])
    # adj = np.array([[[1, 1, 0],
    #                  [0, 1, 0],
    #                  [0, 1, 1]],
    #                 [[1, 1, 0],
    #                  [0, 1, 0],
    #                  [0, 1, 1]]
    #                 ])
    # edges = [[0, 1, 2, 1]]
    # degree = 4
    # print(eval_length(arr, edges, 4))
    # print(evaluator.gst_rsmt(arr2))
    arr = np.random.rand(3, 5, 2)
    edge = [torch.Tensor([0,0,0,0,0]),
            torch.Tensor([1,1,1,1,1]),
            torch.Tensor([0,0,0,0,0])]
