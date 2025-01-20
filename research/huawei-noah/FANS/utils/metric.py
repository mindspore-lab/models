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


from typing import Dict, Union

import numpy as np
import mindspore


class Metric:
    name: str

    @classmethod
    def calculate(cls, candidates: list, ground_truth: list, n, **kwargs):
        pass


class HitRate(Metric):
    name = 'HR'

    @classmethod
    def calculate(cls, candidates: list, ground_truth: list, n, **kwargs):
        candidates_set = set(candidates[:n])
        interaction = candidates_set.intersection(set(ground_truth))
        return int(bool(interaction))


class Recall(Metric):
    name = 'RC'

    @classmethod
    def calculate(cls, candidates: list, ground_truth: list, n, **kwargs):
        candidates_set = set(candidates[:n])
        interaction = candidates_set.intersection(set(ground_truth))
        return len(interaction) * 1.0 / n



class NDCG(Metric):
    name = 'N '

    @staticmethod
    def get_dcg(scores):
        return np.sum(
            np.divide(
                np.power(2, scores) - 1,
                np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)
            ), dtype=np.float32
        )

    @classmethod
    def get_ndcg(cls, rank_list, pos_items):
        relevance = np.ones_like(pos_items)
        it2rel = {it: r for it, r in zip(pos_items, relevance)}
        rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

        idcg = cls.get_dcg(relevance)

        dcg = cls.get_dcg(rank_scores)

        if dcg == 0.0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg

    @classmethod
    def calculate(cls, candidates: list, ground_truth: list, n, **kwargs):
        return cls.get_ndcg(candidates[:n], ground_truth)


class MetricPool:
    def __init__(self):
        self.pool = []
        self.metrics = dict()  # type: Dict[str, Metric]
        self.values = dict()  # type: Dict[tuple, Union[list, float]]
        self.max_n = -1

    def add(self, *metrics: Metric, ns=None):
        ns = ns or [None]

        for metric in metrics:
            self.metrics[metric.name] = metric

            for n in ns:
                if n and n > self.max_n:
                    self.max_n = n
                self.pool.append((metric.name, n))

    def init(self):
        self.values = dict()
        for metric_name, n in self.pool:
            self.values[(metric_name, n)] = []

    def push(self, candidates, ground_truth, **kwargs):
        for metric_name, n in self.values:
            if n and len(ground_truth) < n:
                continue

            self.values[(metric_name, n)].append(self.metrics[metric_name].calculate(
                candidates=candidates,
                ground_truth=ground_truth,
                n=n,
            ))

        # try:
        #     if self.values[(HitRate.name, 5)][-1] < self.values[(HitRate.name, 10)][-1]:
        #         print(candidates)
        #         print(ground_truth)
        #         exit(0)
        # except Exception as e:
        #     print(str(e))
        #     print(self.values[(HitRate.name, 5)])
        #     print(self.values[(HitRate.name, 10)])

    def export(self):
        for metric_name, n in self.values:
            self.values[(metric_name, n)] = mindspore.tensor(
                self.values[(metric_name, n)], dtype=mindspore.float32).mean().item()
