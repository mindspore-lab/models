
import numpy as np
import torch

from utils.rsmt_utils import Evaluator
from utils.rsmt_utils import plot_gst_rsmt
from utils.myutil import plot_directed_graph
from utils.log_utils import *
import matplotlib.pyplot as plt
import argparse

evaluator = Evaluator()


def findx(same_x, all_p, cur_index, edge_list, visited, degree):
    for edge in edge_list:
        u, v = edge
        if u == cur_index or v == cur_index:
            con_p = u if v == cur_index else v
            if visited[con_p] == 0 and all_p[con_p][0] == all_p[cur_index][0]:
                visited[con_p] = 1
                if con_p < degree:
                    same_x.append(con_p)
                else:
                    findx(same_x, all_p, con_p, edge_list, visited, degree)


def findy(same_y, all_p, cur_index, edge_list, visited, degree):
    for edge in edge_list:
        u, v = edge
        if u == cur_index or v == cur_index:
            con_p = u if v == cur_index else v
            if visited[con_p] == 0 and all_p[con_p][1] == all_p[cur_index][1]:
                visited[con_p] = 1
                if con_p < degree:
                    same_y.append(con_p)
                else:
                    findy(same_y, all_p, con_p, edge_list, visited, degree)


def find_noxy(no_xy_u, no_xy_v, all_p, cur_index, edge_list, visited, degree):  # 原生成坐标，生成边序号， 当前sp坐标，当前sp序号
    for edge in edge_list:
        u, v = edge
        if u == cur_index or v == cur_index:
            con_p = u if v == cur_index else v
            if visited[con_p] == 0 and all_p[con_p][0] != all_p[cur_index][0] and all_p[con_p][1] != all_p[cur_index][1]:
                visited[con_p] = 1
                if con_p < degree:
                    if u == con_p:
                        no_xy_u.append(con_p)
                    else:
                        no_xy_v.append(con_p)
                else:
                    if u == con_p:
                        findx(no_xy_u, all_p, con_p, edge_list, visited, degree)
                    if v == con_p:
                        findy(no_xy_v, all_p, con_p, edge_list, visited, degree)


def get_optimal_graph(case, mode='1', getL=False, is_plot=False):
    """
    获取单个样例的最优有向图
    :param case:
    :return:
    """
    src, dst = [], []
    degree = case.shape[0]
    gst_length, sp_list, edge_list = evaluator.gst_rsmt(case)
    # plot_gst_rsmt(case, sp_list, edge_list)
    sp_list = np.array(sp_list)
    if len(sp_list) != 0:
        all_p = np.concatenate([case, sp_list], axis=0)
    sp_num = len(sp_list)
    tmp = set()
    for i, sp in enumerate(sp_list):
        sp_index = i + degree
        visited = np.zeros([degree + sp_num, ])
        same_x, same_y, no_xy_u, no_xy_v = [], [], [], []
        findx(same_x, all_p, sp_index, edge_list, visited, degree)
        findy(same_y, all_p, sp_index, edge_list, visited, degree)
        find_noxy(no_xy_u, no_xy_v, all_p, sp_index, edge_list, visited, degree)
        visited = np.zeros([degree, ])

        for x in same_x:
            for y in same_y:
                if visited[x] == 0 or visited[y] == 0:
                    tmp.add((x, y))

                    visited[x] = 1
                    visited[y] = 1
            for no_v in no_xy_v:
                if visited[x] == 0 or visited[no_v] == 0:
                    tmp.add((x, no_v))

                    visited[x] = 1
                    visited[no_v] = 1
        for no_u in no_xy_u:
            for y in same_y:
                if visited[no_u] == 0 or visited[y] == 0:
                    tmp.add((no_u, y))

                    visited[no_u] = 1
                    visited[y] = 1
    out_adj_table = []
    in_adj_table = []
    adj_table = []
    for _ in range(degree):
        in_adj_table.append(list())
        out_adj_table.append(list())
        adj_table.append(list())

    parents = np.arange(degree, dtype=int)
    def find(x):
        if parents[x] == x:
            return x
        return find(parents[x])

    def union(x, y):
        parents[find(x)] = find(y)

    for x, y in tmp:
        if find(x) != find(y):
            src.append(x)
            dst.append(y)
            in_adj_table[y].append(x)
            out_adj_table[x].append(y)
            adj_table[x].append(y)
            adj_table[y].append(x)
            union(x, y)
    for edge in edge_list:
        if edge[0] < degree and edge[1] < degree:
            src.append(edge[0])
            dst.append(edge[1])
            in_adj_table[edge[1]].append(edge[0])
            out_adj_table[edge[0]].append(edge[1])
            adj_table[edge[1]].append(edge[0])
            adj_table[edge[0]].append(edge[1])

    assert len(src) == degree - 1, '生成错误的有向图，其边数为{}，应为{} \n src:{} \n dst:{}'.format(len(src), degree-1, src, dst)
    for i in range(degree):
        assert i in src or i in dst, '生成错误的有向图, 有环'
    if is_plot:
        fig = plt.figure(1, (15, 4.6))
        plt.subplot(1, 3, 1)
        plt.scatter(case[:, 0], case[:, 1], s=20, c='b', marker=',')
        plt.subplot(1, 3, 2)
        plot_gst_rsmt(case, sp_list, edge_list, 'b', 'b')
        plt.subplot(1, 3, 3)
        plot_directed_graph(case, src, dst, 'b', 'b')
        fig.savefig('../images/con3.png')
    adj = np.eye(degree)
    adj[src, dst] = 1
    if getL:
        if mode == '2':
            return src, dst, gst_length
        elif mode == '3':
            return in_adj_table, out_adj_table, gst_length
        elif mode == '4':
            return adj, in_adj_table, out_adj_table, gst_length
        return adj, gst_length
    if mode == '2':
        return src, dst,
    elif mode == '3':
        return in_adj_table, out_adj_table
    elif mode == '4':
        return adj, in_adj_table, out_adj_table
    return adj


if __name__ == '__main__':
    # case = np.array([[0.0, 0.1], [0.2, 0.2], [0.1, 0.0], [0.3, 0.1]])
    case = np.load('../arr_test.npy')
    src, dst = get_optimal_graph(case, '2', is_plot=True)


