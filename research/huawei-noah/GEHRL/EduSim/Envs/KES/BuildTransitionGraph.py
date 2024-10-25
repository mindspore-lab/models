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
import os
import sys
import warnings
import pickle
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
from EduSim.utils import get_proj_path, get_raw_data_path
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('GEHRL_mindspore')] + 'GEHRL_mindspore')  # 这里要改为你自己的项目的主目录


warnings.filterwarnings('ignore')


# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com

def build_graph(base_data_path, graph_type, dkt_graph_path=None, model_type='GKT'):
    r"""
    Parameters:
        base_data_path: input file path of knowledge tracing data
        graph_type: the type of the concept graph
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    concept_num = 835
    question_list = []
    answer_list = []
    seq_len_list = []
    session_num = 0

    with open(base_data_path, 'r', encoding="utf-8") as f:
        datatxt = f.readlines()
    for index in tqdm(range(len(datatxt)), 'Graph Constructing'):
        line = datatxt[index]
        # 一行是一个二维list，每个元素是[exer_id, 0 or 1] exer_id(0,834)
        one_session_data = json.loads(line)
        if len({log[0] for log in one_session_data}) < 10:
            continue
        question_list.append([el[0] for el in one_session_data])
        answer_list.append([el[1] for el in one_session_data])
        seq_len_list.append(len(one_session_data))
        session_num += 1

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, session_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)

    print(graph)
    np.save(graph, f'{get_proj_path()}/data/dataProcess/junyi/MyTrainsitionGraph.npy')


def build_transition_graph(question_list, seq_len_list, session_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    for i in tqdm(range(session_num), 'trainsition constructing'):
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next_my = questions[j + 1]
            graph[pre, next_my] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1. / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    return graph


def build_dense_graph(node_num):
    graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
    np.fill_diagonal(graph, 0)
    return graph


if __name__ == '__main__':
    base_data_path = f'{get_raw_data_path()}/junyi/student_log_kt_None'
    build_graph(base_data_path, 'Transition')
    num_skills = 835

    graph = np.load(f'{get_proj_path()}/data/dataProcess/junyi/MyTrainsitionGraph.npy')
    # saved_graph = graph.numpy()
    # np.savetxt(f'{get_raw_data_path()}/ASSISTments2015/graph.csv', saved_graph, delimiter=',')

    knowledge_structure = nx.DiGraph()
    bina_graph = np.where(graph > 0.06, 1, 0)
    prerequisite_edges = []
    for i in range(bina_graph.shape[0]):
        for j in range(bina_graph.shape[1]):
            if bina_graph[i, j] == 1:
                if [j, i] in prerequisite_edges:
                    if graph[i, j] > graph[j, i]:
                        prerequisite_edges.append([i, j])
                        prerequisite_edges.remove([j, i])
                    else:
                        continue
                else:
                    prerequisite_edges.append([i, j])
    knowledge_structure.add_nodes_from([i for i in range(num_skills)])
    knowledge_structure.add_edges_from(prerequisite_edges)

    # 去除环
    for cycle in tqdm(list(nx.algorithms.simple_cycles(knowledge_structure)), 'cycle removed'):
        cycle_edges = []
        for i in range(len(cycle) - 1):
            cycle_edges.append([cycle[i], cycle[i + 1], graph[cycle[i], cycle[i + 1]]])
        cycle_edges.append([cycle[len(cycle) - 1], cycle[0], graph[cycle[len(cycle) - 1], cycle[0]]])

        sorted_cycle_edges = sorted(cycle_edges, key=lambda x: x[2])
        if [sorted_cycle_edges[0][0], sorted_cycle_edges[0][1]] in prerequisite_edges:
            prerequisite_edges.remove([sorted_cycle_edges[0][0], sorted_cycle_edges[0][1]])
    # 去除环之后的图
    knowledge_structure = nx.DiGraph()
    knowledge_structure.add_nodes_from([i for i in range(num_skills)])
    knowledge_structure.add_edges_from(prerequisite_edges)

    with open(f"{get_proj_path()}/data/dataProcess/junyi/nxgraph.pkl", "wb") as file:
        str_my = pickle.dumps(knowledge_structure)
        file.write(str_my)

    # 打印topo_order
    _topo_order = list(nx.topological_sort(knowledge_structure))
    print(_topo_order)
    assert not list(nx.algorithms.simple_cycles(knowledge_structure)), "loop in DiGraph"
