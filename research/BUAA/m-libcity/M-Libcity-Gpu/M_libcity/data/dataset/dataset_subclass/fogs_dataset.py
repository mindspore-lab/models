import os
import random

import mindspore.numpy as mnp
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from data.dataset import TrafficStatePointDataset
from gensim.models import Word2Vec
from mindspore import Tensor
# from data.utils import generate_dataloader
from utils import NormalScaler, StandardScaler, MinMax01Scaler, MinMax11Scaler, LogScaler, NoneScaler, \
    ensure_dir


def alias_setup(probs):

    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class TraGraph:
    def __init__(self, nx_G, T, is_directed, p, q, thres=20):
        self.G = nx_G
        self.is_directed = is_directed
        self.T = T
        self.p = p
        self.q = q
        self.thres = thres

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    next = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    count = 0
                    while next != start_node and not self.T[start_node, next] and count < self.thres:
                        count += 1
                        next = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    walk.append(next)
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    count = 0
                    while next != start_node and not self.T[start_node, next] and count < self.thres:
                        count += 1
                        next = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


class FOGSDataset(TrafficStatePointDataset):

    def __init__(self, config):
        super().__init__(config)
        self.y_start = config.get('y_start', 1)
        self.column_wise = config.get('column_wise', False)
        self.thresh_T = config.get('thresh_T', 10)
        self.direct_T = config.get('direct_T', False)
        self.weighted = config.get('weighted', False)
        self.directed = config.get('directed', False)
        self.p = config.get('p', 1)
        self.q = config.get('q', 1)
        self.num_walks = config.get('num_walks', 10)
        self.walk_length = config.get('walk_length', 25)
        self.dimensions = config.get('dimensions', 128)
        self.window_size = config.get('window_size', 10)
        self.workers = config.get('workers', 0)
        self.iter = config.get('iter', 10)
        self.thresh_cos = config.get('thresh_cos', 10)
        self.direct_L = config.get('direct_L', True)
        self.direct = config.get('direct', False)
        self.strides = config.get('strides', 4)
        self.feature_name = {'X': 'float', 'y': 'float', 'x_slot': 'int', 'y_slot': 'int'}
        self.time_volume_mx, self.sim_mx = [], []
        self.dataset_df = self.load_dataset_df()

    def _load_rel(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self.distance_df = relfile.iloc[:, 2:5]

    def construct_learn_mx(self):

        def construct_T(threshold, direct, sim_mx):
            # 用W_V构造T,用knn原理选择每行前threshold为True
            num_nodes = sim_mx.shape[0]
            temporal_graph = np.zeros((num_nodes, num_nodes), dtype=bool)
            for row in range(num_nodes):  # 主对角线为0
                indices = np.argsort(sim_mx[row])[::-1][:threshold]  # 取top k个为True,sim_mx主对角线为0,因此top k不会出现在主对角线上
                temporal_graph[row, indices] = True

            if not direct:  # 构造对称矩阵
                temporal_graph = np.maximum.reduce([temporal_graph, temporal_graph.T])

            return temporal_graph

        def consrtuct_edgelist(distance_df, sensor_ids, weighted=False):
            G = nx.DiGraph()
            sensor_id_to_ind = {}
            for i, sensor_id in enumerate(sensor_ids):
                sensor_id_to_ind[sensor_id] = i

            if weighted:
                for row in distance_df.values:
                    if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
                        continue
                    G.add_edge(sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]], weight=row[2])
            else:
                for row in distance_df.values:
                    if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
                        continue
                    G.add_edge(sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]])
                for edge in G.edges():
                    G[edge[0]][edge[1]]['weight'] = 1
            if not self.directed:
                G = G.to_undirected()
            return G

        def get_cos_similar(v1, v2):
            num = float(np.dot(v1, v2))
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

        def learn_embeddings(walks):

            walks = [list(map(str, walk)) for walk in walks]
            model = Word2Vec(walks, vector_size=self.dimensions, window=self.window_size, min_count=0, sg=1,
                             workers=self.workers,
                             epochs=self.iter)
            return {int(word): model.wv[word] for word in model.wv.index_to_key}

        def learn_final_graph(threshold, direct, embeddings):
            index_list = []
            for index in embeddings.keys():
                index_list.append(index)

            num_nodes = len(index_list)
            cos_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    embedding_i = np.asarray(embeddings[i])
                    embedding_j = np.asarray(embeddings[j])
                    cos_value = get_cos_similar(embedding_i, embedding_j)
                    cos_mx[i][j] = cos_mx[j][i] = cos_value

            learn_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for row in range(num_nodes):
                indices = np.argsort(cos_mx[row])[::-1][:threshold]
                norm = cos_mx[row, indices].sum()
                for index in indices:
                    learn_mx[row, index] = cos_mx[row, index] / norm

            if not direct:
                learn_mx = np.maximum.reduce([learn_mx, learn_mx.T])

            return learn_mx

        def get_adjacency_matrix(distance_df, num_of_vertices, geo_ids):

            A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
            id_dict = {int(i): idx for idx, i in enumerate(geo_ids)}
            df = distance_df
            for row in df.values:
                if len(row) != 3:
                    continue
                i, j = int(row[0]), int(row[1])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
            return A

        def construct_adj_fusion(A, A_dtw, steps):

            N = len(A)
            adj = np.zeros([N * steps] * 2)  # "steps" = 4 !!!

            for i in range(steps):
                if (i == 1) or (i == 2):
                    adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
                else:
                    adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
            # '''
            for i in range(N):
                for k in range(steps - 1):
                    adj[k * N + i, (k + 1) * N + i] = 1
                    adj[(k + 1) * N + i, k * N + i] = 1
            # '''
            adj[3 * N: 4 * N, 0:  N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]
            adj[0: N, 3 * N: 4 * N] = A_dtw  # adj[0 * N : 1 * N, 1 * N : 2 * N]

            adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
            adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
            adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
            adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]

            for i in range(len(adj)):
                adj[i, i] = 1

            return adj

        _, sim_mx = self.get_time_volume_matrix(self.dataset_df)
        dataset_graph_T = construct_T(self.thresh_T, self.direct_T, sim_mx)
        dataset_edgelist = consrtuct_edgelist(self.distance_df, self.geo_ids, self.weighted)
        sparse_matrix = sp.csr_matrix(dataset_graph_T, dtype=bool)
        G = TraGraph(dataset_edgelist, sparse_matrix, self.directed, self.p, self.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(self.num_walks, self.walk_length)
        embeddings = learn_embeddings(walks)
        adj_dtw = learn_final_graph(self.thresh_cos, self.direct_L, embeddings)
        if not self.direct:
            self._logger.info("Use undirected graph")
            adj_dtw = np.maximum.reduce([adj_dtw, adj_dtw.T])
        adj = get_adjacency_matrix(self.distance_df, self.num_nodes, self.geo_ids)
        local_adj = construct_adj_fusion(adj, adj_dtw, steps=self.strides)

        return Tensor(local_adj, mnp.float32)

    def get_data_feature(self):
        """
        返回数据集特征，scaler是归一化方法，adj_mx是邻接矩阵，num_nodes是点的个数，
        feature_dim是输入数据的维度，output_dim是模型输出的维度

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        # 构建 learn mx
        self.adj_mx = self.construct_learn_mx()
        return {"scaler": self.scaler, "adj_mx": self.adj_mx, "ext_dim": self.ext_dim,
                "num_nodes": self.num_nodes, "feature_dim": self.feature_dim,
                "output_dim": self.output_dim, "num_batches": self.num_batches}

    def load_dataset_df(self):
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        df_list = []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            df_list.append(df)
        return np.concatenate(df_list)

    def _generate_data(self):
        """
        加载并生成数据 x, y, x_timeslot, y_timeslot
        @return: x, y, x_timeslot, y_timeslot
        """

        def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
            """
            生成seq2seq样本数据
            :param data: np数据 [B, N, D]
            :param x_offsets:
            :param y_offsets:
            :return:
            """
            num_samples, num_nodes, _ = data.shape
            data = data[:, :, 0:1]  # 只取第一维度的特征

            x, y = [], []
            x_timeslot = []
            y_timeslot = []
            min_t = abs(min(x_offsets))  # 11
            max_t = abs(num_samples - abs(max(y_offsets)))  # num_samples - 12

            for t in range(min_t, max_t):
                x_t = data[t + x_offsets, ...]  # t = 11时, [0,1,2,...,11]
                y_t = data[t + y_offsets, ...]  # t = 11时, [12,13,...,23]

                x.append(x_t)
                y.append(y_t)

                x_timeslot.append((t + x_offsets) % 288)
                y_timeslot.append((t + y_offsets) % 288)

            x = np.stack(x, axis=0)  # [B, T, N ,C]
            y = np.stack(y, axis=0)  # [B ,T, N, C]

            y_timeslot = np.stack(y_timeslot, axis=0)  # (epoch_size, T)
            x_timeslot = np.stack(x_timeslot, axis=0)  # (epoch_size, T)

            return x, y, x_timeslot, y_timeslot

        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        seq_length_x, seq_length_y = self.input_window, self.output_window
        x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
        y_offsets = np.arange(self.y_start, (seq_length_y + 1), 1)
        x_list, y_list, x_timeslot_list, y_timeslot_list = [], [], [], []
        df_list = []
        for filename in data_files:
            df = self._load_dyna(filename)  # (len_time, ..., feature_dim)
            x, y, x_timeslot, y_timeslot = generate_graph_seq2seq_io_data(df, x_offsets, y_offsets)
            x_list.append(x)
            y_list.append(y)
            x_timeslot_list.append(x_timeslot)
            y_timeslot_list.append(y_timeslot)
            df_list.append(df)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        x_timeslot = np.concatenate(x_timeslot_list)
        y_timeslot = np.concatenate(y_timeslot_list)
        df = np.concatenate(df_list)
        return x, y, x_timeslot, y_timeslot, df

    def get_time_volume_matrix(self, df, period=12 * 24 * 7):
        """
        从数据文件中构建时间-流量矩阵
        """
        # 用W_v表示
        data = df[:, :, 0]
        num_samples, num_nodes = data.shape
        num_train = int(num_samples * self.train_rate)
        num_ave = int(num_train / period) * period

        time_volume_mx = np.zeros((num_nodes, 7, 288), dtype=np.float32)
        for node in range(num_nodes):
            for i in range(7):  # 星期一~星期天
                for t in range(288):  # 一天有288个时间段  将所有星期一的288个时间段的流量求均值。同理, 所有星期二, 星期三
                    time_volume = []  # i*288+t表示星期XXX的0点时数据所对应的行数
                    for j in range(i * 288 + t, num_ave, period):
                        time_volume.append(data[j][node])

                    # * modify this line, filter zero values
                    time_volume = np.array(time_volume)
                    time_volume = time_volume[time_volume != 0.]
                    if len(time_volume) != 0:
                        time_volume_mx[node][i][t] = time_volume.mean()
                    else:
                        time_volume_mx[node][i][t] = 0.


        time_volume_mx = time_volume_mx.reshape(num_nodes, -1)  # (num_nodes, 7*288)

        # 计算l2-norm
        similarity_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        similarity_mx[:] = np.inf
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity_mx[i][j] = similarity_mx[j][i] = np.sqrt(
                    np.sum((time_volume_mx[i] - time_volume_mx[j]) ** 2))

        distances = similarity_mx[~np.isinf(similarity_mx)].flatten()
        std = distances.std()
        similarity_mx = np.exp(-np.square(similarity_mx / std))  # 主对角线为0

        return time_volume_mx, similarity_mx

    def split_train_val_test(self, x, y, x_timeslot, y_timeslot, df):
        """
        切分数据 train val test
        @param x:
        @param y:
        @param x_timeslot:
        @param y_timeslot:
        @return: x_train, y_train, x_timeslot_train, y_timeslot_train, x_val, y_val, x_timeslot_val, y_timeslot_val,\
            x_test, y_test, x_timeslot_test, y_timeslot_test
        """
        # 处理均值数据
        time_volume_mx, _ = self.get_time_volume_matrix(df)
        num_nodes, timeslot = time_volume_mx.shape
        pickle_mean_data = np.reshape(time_volume_mx, (num_nodes, 7, 288))
        pickle_mean_data = np.mean(pickle_mean_data, axis=1)  # (num_nodes, 288)  # 每个结点在288个时间段的均值

        # 划分
        num_samples = x.shape[0]
        num_test = round(num_samples * (1 - self.train_rate - self.eval_rate))
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_train - num_test

        # 训练集
        x_train, y_train = x[:num_train], y[:num_train]
        x_timeslot_train = x_timeslot[:num_train]
        y_timeslot_train = x_timeslot[:num_train]  # (len, 12)

        # 验证集
        x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
        y_timeslot_val = y_timeslot[num_train: num_train + num_val]
        x_timeslot_val = x_timeslot[num_train: num_train + num_val]

        # 测试集
        x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]
        y_timeslot_test = y_timeslot[-num_test:]
        x_timeslot_test = x_timeslot[-num_test:]

        constant = 5
        # 改训练集的标签，转换为趋势
        x_train_len, T, num_nodes, input_dim = x_train.shape  # (len, 12, num_nodes, 1)
        for index in range(x_train_len):
            x_train_value = x_train[index][T - 1]
            cur_timeslot = x_timeslot_train[index][-1]  # 最后一个时刻对应的时间段, 整数
            indices = []
            for i in range(num_nodes):  # 确定x_train_val为0值的元素所对应的索引
                if x_train_value[i, 0] == 0:
                    indices.append(i)

            for ind in indices:
                for t in range(T - 1)[::-1]:  # 把x_train_val为0值所对应的元素往前替换为最新的那个
                    if x_train[index][t][ind][0] != 0:
                        x_train_value[ind][0] = x_train[index][t][ind][0]
                        break

            # 如果前面11个数据为0, 则用当前时间段及之前的均值替换0
            for ind in indices:
                if x_train_value[ind, 0] == 0:
                    for prev_timeslot in range(cur_timeslot + 1)[::-1]:
                        if pickle_mean_data[ind][prev_timeslot] != 0:  # 如果当前时间段均值为0,则往前寻找非0均值
                            x_train_value[ind][0] = pickle_mean_data[ind][cur_timeslot]
                            break

            # 如果均值也还是0的话,用常数代替
            for ind in indices:
                if x_train_value[ind, 0] == 0:
                    x_train_value[ind][0] = constant

            for t in range(T):
                for node in range(num_nodes):
                    y_train[index][t][node][0] = (y_train[index][t][node][0] - x_train_value[node][0]) / \
                                                 x_train_value[node][0]

        # 改变验证集和测试集的训练数据
        x_val_len, T, num_nodes, input_dim = x_val.shape  # (len, 12, num_nodes, 1)
        for index in range(x_val_len):
            x_val_value = x_val[index][T - 1]
            cur_timeslot = x_timeslot_val[index][-1]  # 最后一个时刻对应的时间段, 整数
            indices = []
            for i in range(num_nodes):  # 确定x_val_value为0值的元素所对应的索引
                if x_val_value[i, 0] == 0:
                    indices.append(i)

            for ind in indices:
                for t in range(T - 1)[::-1]:
                    if x_val[index][t][ind][0] != 0:
                        x_val_value[ind][0] = x_val[index][t][ind][0]
                        break

            # 如果前面11个数据为0, 则用当前时间段及之前的均值替换0
            for ind in indices:
                if x_val_value[ind, 0] == 0:
                    for prev_timeslot in range(cur_timeslot + 1)[::-1]:
                        if pickle_mean_data[ind][prev_timeslot] != 0:
                            x_val_value[ind][0] = pickle_mean_data[ind][cur_timeslot]
                            break

            # 如果均值也还是0的话,用常数代替
            for ind in indices:
                if x_val_value[ind, 0] == 0:
                    x_val_value[ind][0] = constant

        x_test_len, T, num_nodes, input_dim = x_test.shape  # (len, 12, num_nodes, 1)
        for index in range(x_test_len):
            x_test_value = x_test[index][T - 1]
            cur_timeslot = x_timeslot_test[index][-1]  # 最后一个时刻对应的时间段, 整数
            indices = []
            for i in range(num_nodes):  # 确定x_val_value为0值的元素所对应的索引
                if x_test_value[i, 0] == 0:
                    indices.append(i)

            for ind in indices:
                for t in range(T - 1)[::-1]:
                    if x_test[index][t][ind][0] != 0:
                        x_test_value[ind][0] = x_test[index][t][ind][0]
                        break

            # 如果前面11个数据为0, 则用当前时间段及之前的均值替换0
            for ind in indices:
                if x_test_value[ind, 0] == 0:
                    for prev_timeslot in range(cur_timeslot + 1)[::-1]:
                        if pickle_mean_data[ind][prev_timeslot] != 0:
                            x_test[ind][0] = pickle_mean_data[ind][cur_timeslot]
                            break

            # 如果均值也还是0的话,用常数代替
            for ind in indices:
                if x_test_value[ind, 0] == 0:
                    x_test_value[ind][0] = constant

        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape) +
                          ", x_timeslot_train: " + str(x_timeslot_train.shape) +
                          ", y_timeslot_train: " + str(y_timeslot_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape) +
                          ", x_timeslot_val: " + str(x_timeslot_val.shape) +
                          ", y_timeslot_val: " + str(y_timeslot_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape) +
                          ", x_timeslot_test: " + str(x_timeslot_test.shape) +
                          ", y_timeslot_test: " + str(y_timeslot_test.shape))

        if self.cache_dataset:
            ensure_dir(self.cache_file_folder)
            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_timeslot_train=x_timeslot_train,
                y_timeslot_train=y_timeslot_train,
                x_val=x_val,
                y_val=y_val,
                x_timeslot_val=x_timeslot_val,
                y_timeslot_val=y_timeslot_val,
                x_test=x_test,
                y_test=y_test,
                x_timeslot_test=x_timeslot_test,
                y_timeslot_test=y_timeslot_test
            )
            self._logger.info('Saved at ' + self.cache_file_name)

        return x_train, y_train, x_timeslot_train, y_timeslot_train, x_val, y_val, x_timeslot_val, y_timeslot_val, \
               x_test, y_test, x_timeslot_test, y_timeslot_test

    def _generate_train_val_test(self):

        x, y, x_timeslot, y_timeslot, df = self._generate_data()
        self.data_df = df
        return self.split_train_val_test(x, y, x_timeslot, y_timeslot, df)

    def _get_scalar(self, scaler_type, x_train, y_train):

        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max()))
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max()), minn=min(x_train.min()))
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max()), minn=min(x_train.min()))
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def _load_cache_train_val_test(self):

        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_timeslot_train = cat_data['x_timeslot_train']
        y_timeslot_train = cat_data['y_timeslot_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_timeslot_test = cat_data['x_timeslot_test']
        y_timeslot_test = cat_data['y_timeslot_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        x_timeslot_val = cat_data['x_timeslot_val']
        y_timeslot_val = cat_data['y_timeslot_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape) +
                          ", x_timeslot_train: " + str(x_timeslot_train.shape) +
                          ", y_timeslot_train: " + str(y_timeslot_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape) +
                          ", x_timeslot_val: " + str(x_timeslot_val.shape) +
                          ", y_timeslot_val: " + str(y_timeslot_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape) +
                          ", x_timeslot_test: " + str(x_timeslot_test.shape) +
                          ", y_timeslot_test: " + str(y_timeslot_test.shape))
        return x_train, y_train, x_timeslot_train, y_timeslot_train, x_val, y_val, x_timeslot_val, y_timeslot_val, \
               x_test, y_test, x_timeslot_test, y_timeslot_test

    def get_data(self):
        x_train, y_train, x_timeslot_train, y_timeslot_train, x_val, y_val, x_timeslot_val, y_timeslot_val, \
        x_test, y_test, x_timeslot_test, y_timeslot_test = [], [], [], [], [], [], [], [], [], [], [], []
        # 生成/加载 train val test 数据
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_timeslot_train, y_timeslot_train, x_val, y_val, x_timeslot_val, y_timeslot_val, \
                x_test, y_test, x_timeslot_test, y_timeslot_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_timeslot_train, y_timeslot_train, x_val, y_val, x_timeslot_val, y_timeslot_val, \
                x_test, y_test, x_timeslot_test, y_timeslot_test = self._generate_train_val_test()
        self.feature_dim = x_train.shape[-1]
        # 获取归一化的方式
        self.scaler = self._get_scalar(self.scaler_type, x_train, y_train)
        # 归一化 只对 x 做归一化
        x_train[..., 0] = self.scaler.transform(x_train[..., 0])
        x_val[..., 0] = self.scaler.transform(x_val[..., 0])
        x_test[..., 0] = self.scaler.transform(x_test[..., 0])
        # 转 dataloader
        train_data = list(zip(x_train, y_train, x_timeslot_train, y_timeslot_train))
        eval_data = list(zip(x_val, y_val, x_timeslot_val, y_timeslot_val))
        test_data = list(zip(x_test, y_test, x_timeslot_test, y_timeslot_test))
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            generate_dataloader(train_data, eval_data, test_data, self.feature_name,
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader


# 补充utils
import mindspore.dataset as ds
from mindspore.communication.management import get_rank  # 多卡
from data.batch import Batch


def l2l(origin_len):

    loc_len = origin_len['current_loc']
    history_len = origin_len['history_loc']
    return loc_len, history_len


def padding(pad_item, pad_max_len, feature_name, data):

    pad_len = {}
    f2indx = {}
    origin_len = {}

    pad_max_len = pad_max_len if pad_max_len is not None else {}
    if pad_max_len == {} and "history_loc" in pad_item:
        for key in pad_item:
            pad_max_len[key] = 128

    pad_item = pad_item if pad_item is not None else {}
    for indx, key in enumerate(feature_name):
        f2indx[key] = indx
        if key in pad_item:
            pad_len[key] = 0
            origin_len[key] = []

    for i in range(len(data)):
        for indx, key in enumerate(pad_item):
            findx = f2indx[key]
            # 需保证 item 每个特征的顺序与初始化时传入的 feature_name 中特征的顺序一致
            origin_len[key].append(len(data[i][findx]))
            if pad_len[key] < len(data[i][findx]):
                # 保持 pad_len 是最大的
                pad_len[key] = len(data[i][findx])

    for indx, key in enumerate(pad_item):
        # 只对在 pad_item 中的特征进行补齐
        findx = f2indx[key]
        max_len = pad_len[key]
        if key in pad_max_len:
            max_len = min(pad_max_len[key], max_len)
        for i in range(len(data)):
            if len(data[i][findx]) < max_len:
                data[i][findx] += [pad_item[key]] * \
                                  (max_len - len(data[i][findx]))
            else:
                # 截取的原则是，抛弃前面的点
                data[i][findx] = data[i][findx][-max_len:]
                origin_len[key][i] = max_len

    if "current_loc" in pad_item:
        loc_len, his_len = l2l(origin_len)
        for i in range(len(data)):
            data[i].append(loc_len[i])
            data[i].append(his_len[i])

    return data


class ListDataset:
    def __init__(self, data, feature_name):
        self.data = data
        self.feature_name = feature_name

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True,
                        pad_with_last_sample=False, rank_size=1):
    if os.getenv('RANK_SIZE'):
        rank_size = int(os.getenv('RANK_SIZE'))
        try:
            assert len(train_data) > (rank_size * batch_size)
            assert len(eval_data) > (rank_size * batch_size)
            assert len(test_data) > (rank_size * batch_size)
        except Exception:
            batch_size = 16
            print('Step size is lower than rank size, change the Batch Size to 16.')

    train_data_me = []
    for i1 in range(len(train_data)):
        tuple_me = ()
        for i2 in range(len(train_data[i1])):
            ndarry_me = train_data[i1][i2].astype(np.float32)
            tuple_me = tuple_me + (ndarry_me,)
        train_data_me.append(tuple_me)
    eval_data_me = []
    for i1 in range(len(eval_data)):
        tuple_me = ()
        for i2 in range(len(eval_data[i1])):
            ndarry_me = eval_data[i1][i2].astype(np.float32)
            tuple_me = tuple_me + (ndarry_me,)
        eval_data_me.append(tuple_me)
    test_data_me = []
    for i1 in range(len(test_data)):
        tuple_me = ()
        for i2 in range(len(test_data[i1])):
            ndarry_me = test_data[i1][i2].astype(np.float32)
            tuple_me = tuple_me + (ndarry_me,)
        test_data_me.append(tuple_me)

    train_dataset = ListDataset(train_data_me, feature_name)
    eval_dataset = ListDataset(eval_data_me, feature_name)
    test_dataset = ListDataset(test_data_me, feature_name)

    _feature_names = list(feature_name.keys())
    _feature_types = list(feature_name.values())

    def collaensure_dirtor(X, y, x_slot, y_slot, BatchInfo):

        feature_name = {
            'X': type(X),
            'y': type(y),
            'x_slot': type(x_slot),
            'y_slot': type(y_slot),
        }
        batch = Batch(feature_name)

        num_samples = len(X)
        for i in range(num_samples):
            sample = {'X': X[i], 'y': y[i], 'x_slot': x_slot[i], 'y_slot': y_slot[i]}
            batch.append(list(sample.values()))

        batch_data = {
            'X': batch['X'],
            'y': batch['y'],
            'x_slot': batch['x_slot'],
            'y_slot': batch['y_slot']
        }
        return batch_data['X'], batch_data['y'], batch_data['x_slot'], batch_data['y_slot']

    train_dataloader, eval_dataloader, test_dataload = None, None, None

    if rank_size > 1:
        rank_id = get_rank()

        train_dataloader = ds.GeneratorDataset(source=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle, num_shards=rank_size, shard_id=rank_id
                                               ).batch(batch_size, drop_remainder=True,
                                                       per_batch_map=collaensure_dirtor)

        eval_dataloader = ds.GeneratorDataset(source=eval_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)

        test_dataloader = ds.GeneratorDataset(source=test_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)
    else:

        train_dataloader = ds.GeneratorDataset(source=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle).batch(batch_size, drop_remainder=True,
                                                                      per_batch_map=collaensure_dirtor)
        eval_dataloader = ds.GeneratorDataset(source=eval_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)
        test_dataloader = ds.GeneratorDataset(source=test_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)

    return train_dataloader, eval_dataloader, test_dataloader


def generate_dataloader_pad(train_data, eval_data, test_data, feature_name,
                            batch_size, num_workers, pad_item=None,
                            pad_max_len=None, shuffle=True, rank_size=1):
    if os.getenv('RANK_SIZE'):
        rank_size = int(os.getenv('RANK_SIZE'))

    train_dataset = ListDataset(train_data, feature_name)
    eval_dataset = ListDataset(eval_data, feature_name)
    test_dataset = ListDataset(test_data, feature_name)

    train_dataset = padding(pad_item, pad_max_len, feature_name, data=train_dataset)
    eval_dataset = padding(pad_item, pad_max_len, feature_name, data=eval_dataset)
    test_dataset = padding(pad_item, pad_max_len, feature_name, data=test_dataset)

    _feature_names = list(feature_name.keys())

    if "current_loc" in pad_item:
        _feature_names.append('loc_len')
        _feature_names.append('his_len')

    _feature_types = list(feature_name.values())

    def collaensure_dirtor(X, y, x_slot, y_slot, BatchInfo):
        feature_name = {
            'X': type(X),
            'y': type(y),
            'x_slot': type(x_slot),
            'y_slot': type(y_slot),
        }
        batch = Batch(feature_name)

        num_samples = len(X)
        for i in range(num_samples):
            sample = {'X': X[i], 'y': y[i], 'x_slot': x_slot[i], 'y_slot': y_slot[i]}
            batch.append(list(sample.values()))

        batch_data = {
            'X': batch['X'],
            'y': batch['y'],
            'x_slot': batch['x_slot'],
            'y_slot': batch['y_slot']
        }
        return batch_data['X'], batch_data['y'], batch_data['x_slot'], batch_data['y_slot']

    train_dataloader, eval_dataloader, test_dataload = None, None, None
    if rank_size > 1:
        rank_id = get_rank()

        train_dataloader = ds.GeneratorDataset(source=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle, num_shards=rank_size, shard_id=rank_id
                                               ).batch(batch_size, drop_remainder=True,
                                                       per_batch_map=collaensure_dirtor)

        eval_dataloader = ds.GeneratorDataset(source=eval_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)

        test_dataloader = ds.GeneratorDataset(source=test_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)
    else:

        train_dataloader = ds.GeneratorDataset(source=train_dataset, column_names=_feature_names,
                                               num_parallel_workers=num_workers,
                                               shuffle=shuffle).batch(batch_size, drop_remainder=True,
                                                                      per_batch_map=collaensure_dirtor)
        eval_dataloader = ds.GeneratorDataset(source=eval_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers,
                                              shuffle=shuffle).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)
        test_dataloader = ds.GeneratorDataset(source=test_dataset, column_names=_feature_names,
                                              num_parallel_workers=num_workers
                                              , shuffle=False).batch(batch_size, drop_remainder=True,
                                                                     per_batch_map=collaensure_dirtor)

    return train_dataloader, eval_dataloader, test_dataloader
