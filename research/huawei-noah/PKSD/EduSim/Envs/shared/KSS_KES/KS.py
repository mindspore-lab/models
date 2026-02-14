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
# -*- coding:utf-8 _*-

import networkx as nx


class KS(nx.DiGraph):
    def dump_id2idx(self, filename):
        with open(filename, "w") as wf:
            for node in self.nodes:
                print("%s,%s" % (node, node), file=wf)

    def dump_graph_edges(self, filename):
        with open(filename, "w") as wf:
            for edge in self.edges:
                print("%s,%s" % edge, file=wf)


def bfs(graph, mastery, pnode, hop, candidates, soft_candidates, visit_nodes=None, visit_threshold=1,
        allow_shortcut=True):  # pragma: no cover
    # bfs(graph, mastery, node, 1, candidates, soft_candidates, visit_nodes, visit_threshold, allow_shortcut)
    assert hop >= 0
    if visit_nodes and visit_nodes.get(pnode, 0) >= visit_threshold:
        return

    if allow_shortcut is False or mastery[pnode] < 0.5:
        candidates.add(pnode)
    else:
        soft_candidates.add(pnode)

    if hop == 0:
        return

    # 向前搜索
    for node in list(graph.predecessors(pnode)):
        if allow_shortcut is False or mastery[node] < 0.5:
            bfs(
                graph=graph,
                mastery=mastery,
                pnode=node,
                hop=hop - 1,
                candidates=candidates,
                soft_candidates=soft_candidates,
                visit_nodes=visit_nodes,
                visit_threshold=visit_threshold,
                allow_shortcut=allow_shortcut,
            )

    # 向后搜索
    for node in list(graph.successors(pnode)):
        if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
            continue
        if allow_shortcut is False or mastery[node] < 0.5:
            candidates.add(node)
        else:
            soft_candidates.add(node)


def influence_control(graph, mastery, pnode, visit_nodes=None, visit_threshold=1, allow_shortcut=True, no_pre=None,
                      connected_graph=None, target=None, legal_candidates=None,
                      path_table=None) -> tuple:  # pragma: no cover
    """

    Parameters
    ----------
    graph: nx.Digraph
    mastery: list(float)
    pnode: None or int
    visit_nodes: None or dict
    visit_threshold: int
    allow_shortcut: bool
    no_pre: set
    connected_graph: dict
    target: set or list
    legal_candidates: set or None
    path_table: dict or None

    Returns
    -------

    """
    assert pnode is None or isinstance(pnode, int), pnode

    if mastery is None:
        allow_shortcut = False

    # select candidates
    candidates = []
    soft_candidates = []

    if allow_shortcut is True:
        # 允许通过捷径绕过已掌握的点

        # 在已有前驱节点前提下，如果当前节点已经掌握，那么开始学习它的后继未掌握节点
        if pnode is not None and mastery[pnode] >= 0.5:
            for candidate in list(graph.successors(pnode)):
                if visit_nodes and visit_nodes.get(candidate, 0) >= visit_threshold:
                    continue
                if mastery[candidate] < 0.5:
                    candidates.append(candidate)
                else:
                    soft_candidates.append(candidate)
            if candidates:
                return candidates, soft_candidates

        # 否则(即当前节点未掌握), 选取其2跳前驱点及所有前驱点的后继点（未掌握的）作为候选集
        elif pnode is not None:
            _candidates = set()
            _soft_candidates = set()
            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 2, _candidates, _soft_candidates, visit_nodes, visit_threshold,
                    allow_shortcut)
            return list(_candidates) + [pnode], list(_soft_candidates)

        # 如果前两种方法都没有选取到候选集，那么进行重新选取
        for node in graph.nodes:
            if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
                # 当前结点频繁访问
                continue

            if mastery[node] >= 0.5:
                # 当前结点已掌握，跳过
                soft_candidates.append(node)
                continue

            # 当前结点未掌握，且其前置点都掌握了的情况下，加入候选集
            pre_nodes = list(graph.predecessors(node))
            for n in pre_nodes:
                pre_mastery = mastery[n]
                if pre_mastery < 0.5:
                    soft_candidates.append(node)
                    break
            else:
                candidates.append(node)
    else:
        # allow_shortcut is False
        # 不允许通过捷径绕过已掌握的点
        candidates = set()
        soft_candidates = set()
        if pnode is not None:
            # 加入所有后继点
            candidates = set(list(graph.successors(pnode)))

            if not graph.predecessors(pnode) or not graph.successors(pnode):
                # 没有前驱点 或 没有后继点
                candidates = set(no_pre)

            # 选取其2跳前驱点及所有1跳前驱点的后继点
            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 1, candidates, soft_candidates, visit_nodes, visit_threshold, allow_shortcut)

            # 避免死循环
            if candidates:
                candidates.add(pnode)

            # 频繁访问节点过滤
            if visit_nodes:
                candidates -= set([node for node, count in visit_nodes.items() if count >= visit_threshold])

            candidates = list(candidates)

    if not candidates:
        # 规则没有选取到合适候选集
        candidates = list(graph.nodes)
        soft_candidates = list()

    if connected_graph is not None and pnode is not None:
        # 保证候选集和pnode在同一个连通子图内
        candidates = list(set(candidates) & connected_graph[pnode])

    if target is not None and legal_candidates is not None:
        assert target
        # 保证节点可达目标点
        _candidates = set(candidates) - legal_candidates
        for candidate in _candidates:
            if candidate in legal_candidates:
                continue
            for t in target:
                if path_table is not None:
                    if t in path_table[candidate]:
                        legal_tag = True
                    else:
                        legal_tag = False
                else:
                    legal_tag = nx.has_path(graph, candidate, t)
                if legal_tag is True:
                    legal_candidates.add(candidate)
                    break
        candidates = set(candidates) & legal_candidates
        if not candidates:
            candidates = target

    return list(candidates), list(soft_candidates)
