# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import random

import sys
import os
import warnings
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from EduSim.utils import get_proj_path, get_raw_data_path
from EduSim.Envs.KES import KESEnv
from EduSim.Envs.KES_ASSIST15 import KESASSISTEnv

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('GEHRL_mindspore')] + 'GEHRL_mindspore')  # 这里要改为你自己的项目的主目录
sys.path.append(os.path.dirname(sys.path[0]))  # 这里要改为你自己的项目的主目录

# 忽略warning的输出
warnings.filterwarnings('ignore')


class Node2v:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if cur_nbrs:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next_my = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next_my)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        for _ in range(num_walks):
            # print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
                if random.random() < 0.3:
                    print(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
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
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while smaller and larger:
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
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return_thing = kk
    else:
        return_thing = J[kk]
    return return_thing


def deepwalk_walk(g, walk_length=40, start_node=None):
    walks = [start_node]
    while len(walks) < walk_length:
        cur = walks[-1]
        cur_nbs = list(g.neighbors(cur))
        if cur_nbs:
            walks.append(random.choice(cur_nbs))
        else:
            break
            # raise ValueError('node with 0 in_degree')
    return walks


def sample_walks(g, walk_length=40, number_walks=10):
    total_walks = []
    # print('Start sampling walks:')
    nodes = list(g.nodes())
    for _ in range(number_walks):
        # print(f'\t iter:{iter_ + 1}/{number_walks}')
        random.shuffle(nodes)
        for node in nodes:
            total_walks.append(deepwalk_walk(g, walk_length, node))
            if random.random() < 0.3:
                print(deepwalk_walk(g, walk_length, node))
    return total_walks


def save_as_numpy(graph, word2vec_model, np_save_path):
    result = []
    for i in range(len(graph.nodes)):
        result.append(word2vec_model.wv[i])
    result = np.array(result)
    np.save(np_save_path, result)
    print('numpy saved')


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # args = sys.argv[1]
    print('Graph embedding start...')
    # 环境设定
    # envKSS = KSSEnv()

    dataRecPath = f'{get_proj_path()}/data/dataProcess/junyi/dataRec'
    envKES = KESEnv(dataRec_path=dataRecPath)

    dataRecPath = f'{get_raw_data_path()}/ASSISTments2015/processed/'
    envKESASSIST = KESASSISTEnv(dataRec_path=dataRecPath)

    # dataRecPath = f'{get_raw_data_path()}/ASSISTments2009/assist09.npz'
    # envKESASSIST09 = KESASSIST09Env(dataRec_path=dataRecPath)

    # 环境选择
    """这4个参数是关键"""
    embed_type = 'node2vec'
    is_directed = True
    test_flag = False
    # graph = envKES.knowledge_structure
    graph = envKESASSIST.knowledge_structure

    # if args == '1':
    #     graph = envKSS.knowledge_structure
    # elif args == '2':
    #     graph = envKES.knowledge_structure
    # elif args == '3':
    #     graph = envKESASSIST.knowledge_structure
    # elif args == '4':
    #     graph = envKESASSIST09.knowledge_structure
    # else:
    #     raise ValueError('Incorrect dataset number!')
    """"""
    if not is_directed:
        graph = graph.to_undirected()  # 有向图变无向图
    for edge in graph.edges():
        u, v = edge[0], edge[1]
        graph[u][v]['weight'] = 1.0

    # 参数设置
    walk_length = None
    number_walks = None
    emb_size = None
    window_size = None
    model_save_file_path = ''
    embedding_save_file_path = ''
    print(len(graph.nodes))
    if len(graph.nodes) == 835:  # junyi
        walk_length = 100
        number_walks = 100
        emb_size = 100
        window_size = 5

        dir_my = f'{get_proj_path()}/EduSim/Envs/meta_data/'
        os.makedirs(dir_my, exist_ok=True)
        model_save_file_path = f'{get_proj_path()}/EduSim/Envs/meta_data/junyiGraphEmbedding.ckpt'
        np_save_path = f'{dir_my}/junyiGraphEmbedding.npy'
    elif len(graph.nodes) == 100:  # ASSISTments15
        walk_length = 100  # 100
        number_walks = 100  # 100
        emb_size = 80  # 80
        window_size = 5

        model_save_file_path = f'{get_proj_path()}/EduSim/Envs/meta_data/ASSISTwvModel.ckpt'
        # embedding_save_file_path = f'{get_proj_path()}/EduSim/Envs/meta_data/ASSISTGraphEmbedding.ckpt'
        np_save_path = f'{get_proj_path()}/EduSim/Envs/meta_data/ASSISTGraphEmbedding.npy'
    else:
        raise ValueError('Wrong Graph')

    # 得到embedding
    if not test_flag:
        total_walks = []
        if embed_type == 'deepwalk':
            total_walks = sample_walks(graph, walk_length=walk_length, number_walks=number_walks)
        elif embed_type == 'node2vec':
            embedding_model = Node2v(graph, is_directed=is_directed, p=0.25, q=0.5)
            embedding_model.preprocess_transition_probs()
            total_walks = embedding_model.simulate_walks(walk_length=walk_length, num_walks=number_walks)

        model_w2v = Word2Vec(sentences=total_walks, sg=0, hs=0, vector_size=emb_size, window=window_size,
                             min_count=0, workers=10, epochs=10)
        """sg : {0, 1}, optional
                Training algorithm: 1 for skip-gram; otherwise CBOW.
            hs : {0, 1}, optional
                If 1, hierarchical softmax will be used for model training.
                If 0, and `negative` is non-zero, negative sampling will be used."""
        # 保存结果
        model_w2v.save(model_save_file_path)
        save_as_numpy(graph, model_w2v, np_save_path)
        print(f'result saved at {model_save_file_path}')

    # 测试
    model_test = Word2Vec.load(model_save_file_path)
    # save_as_numpy(graph, model_test, np_save_path)

    source_node = 5
    print(model_test.wv.most_similar(source_node, topn=9))

    pre_nodes = list(nx.bfs_tree(graph, source_node, reverse=True).nodes())
    suc_nodes = list(nx.bfs_tree(graph, source_node, reverse=False).nodes())
    pre_graph = graph.subgraph(pre_nodes)
    suc_graph = graph.subgraph(suc_nodes)
    pre_topo_order = list(nx.topological_sort(pre_graph))
    suc_topo_order = list(nx.topological_sort(suc_graph))
    print(f'pre_nodes: {pre_topo_order}')
    print(f'suc_nodes: {suc_topo_order}')
    print(f'total pre num: {len(pre_nodes)}')
    print(f'total suc num: {len(suc_nodes)}')

    max_value = 0
    for i in range(len(graph.nodes)):
        for j in range(len(model_test.wv[i])):
            if abs(model_test.wv[i][j]) > max_value:
                max_value = abs(model_test.wv[i][j])
    print(f'Values in range [-{max_value}, {max_value}]')  # KSS: 4, KESjunyi:4, KESAssis: 2
