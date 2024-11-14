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

def load_dataset(base_data_path, graph_type, dkt_graph_path=None, model_type='GKT'):
    r"""
    Parameters:
        base_data_path: input file path of knowledge tracing data
        graph_type: the type of the concept graph
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    concept_num = 100
    question_list = []
    answer_list = []
    seq_len_list = []
    student_num = 0

    for k, _ in enumerate(os.listdir(base_data_path)):
        for _, file_name in tqdm(enumerate(os.listdir(f'{base_data_path}{k + 1}/train/'))):
            with open(f'{base_data_path}{k + 1}/train/{file_name}', 'r') as f:
                one_student_data = f.readlines()[1:]  # header exists
                # one_student_data = one_student_data[:200]
            # [[exer_id, 0 or 1]...] 0=<exer_id<=99
            one_student_data = [[int(line.rstrip().split(',')[0]) - 1, int(line.rstrip().split(',')[1])]
                                for _, line in enumerate(one_student_data)]
            question_list.append([el[0] for el in one_student_data])
            answer_list.append([el[1] for el in one_student_data])
            seq_len_list.append(len(one_student_data))
            student_num += 1

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        elif graph_type == 'Support':
            graph = build_support_graph(question_list, answer_list, seq_len_list, student_num)

    print(graph)
    np.save(f'{get_proj_path()}/data/dataProcess/ASSISTments2015/MyTransitionGraph.npy', graph)


def build_support_graph(question_list, answer_list, seq_len_list, student_num):
    # graph = np.zeros((concept_num, concept_num))
    concept_num = 100
    count_i_j_1_1 = np.zeros((concept_num, concept_num))
    count_i_j_1_0 = np.zeros((concept_num, concept_num))
    count_i_j_0_0 = np.zeros((concept_num, concept_num))
    count_i_j_0_1 = np.zeros((concept_num, concept_num))

    for k in tqdm(range(student_num)):
        questions = question_list[k]
        answers = answer_list[k]
        seq_len = seq_len_list[k]

        for m in range(seq_len - 1):
            pre = questions[m]
            for n in range(m + 1, seq_len - 1):
                next_my = questions[n]

                if answers[m] == 1 and answers[n] == 1:
                    count_i_j_1_1[pre, next_my] += 1
                elif answers[m] == 1 and answers[n] == 0:
                    count_i_j_1_0[pre, next_my] += 1
                elif answers[m] == 0 and answers[n] == 1:
                    count_i_j_0_1[pre, next_my] += 1
                elif answers[m] == 0 and answers[n] == 0:
                    count_i_j_0_0[pre, next_my] += 1
    np.fill_diagonal(count_i_j_1_1, 0)
    np.fill_diagonal(count_i_j_1_0, 0)
    np.fill_diagonal(count_i_j_0_1, 0)
    np.fill_diagonal(count_i_j_0_0, 0)

    # p_Ri_Rj = np.zeros((concept_num, concept_num))
    # p_Ri_Rj_Wj = np.zeros((concept_num, concept_num))
    # p_Wj_Wi = np.zeros((concept_num, concept_num))
    # P_Wj_Ri_Wi = np.zeros((concept_num, concept_num))

    p_Ri_Rj = (count_i_j_1_1.T + 0.01) / (count_i_j_1_0.T + count_i_j_1_1.T + 0.01)
    p_Ri_Rj_Wj = ((count_i_j_0_1.T + count_i_j_1_1.T + 0.01) /
                  (count_i_j_0_1.T + count_i_j_1_1.T + count_i_j_1_0.T + count_i_j_0_0.T + 0.01))
    p_Wj_Wi = (count_i_j_0_0 + 0.01) / (count_i_j_0_0 + count_i_j_0_1 + 0.01)
    p_Wj_Ri_Wi = ((count_i_j_1_0 + count_i_j_0_0 + 0.01) /
                  (count_i_j_0_1 + count_i_j_1_1 + count_i_j_1_0 + count_i_j_0_0 + 0.01))

    graph = np.maximum(np.log(p_Ri_Rj / p_Ri_Rj_Wj), 0) + np.maximum(np.log(p_Wj_Wi / p_Wj_Ri_Wi), 0)

    return graph


def build_transition_graph(question_list, seq_len_list, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    for i in range(student_num):
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
    base_data_path = f'{get_raw_data_path()}/ASSISTments2015/processed/'
    load_dataset(base_data_path, 'Transition')
    num_skills = 100

    graph = np.load(f'{get_proj_path()}/data/dataProcess/ASSISTments2015/MyTransitionGraph.npy')

    # saved_graph = graph.numpy()
    # np.savetxt(f'{get_raw_data_path()}/ASSISTments2015/graph.csv', saved_graph, delimiter=',')

    knowledge_structure = nx.DiGraph()
    bina_graph = np.where(graph > 0.09, 1, 0)
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

    with open(f"{get_proj_path()}/data/dataProcess/ASSISTments2015/nxgraph.pkl", "wb") as file:
        str_my = pickle.dumps(knowledge_structure)
        file.write(str_my)

    # 打印topo_order
    _topo_order = list(nx.topological_sort(knowledge_structure))
    print(_topo_order)
    assert not list(nx.algorithms.simple_cycles(knowledge_structure)), "loop in DiGraph"
