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


import mindspore
from loader.task.utils.base_cluster_mlm_task import BaseClusterMLMTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask, CurriculumMLMBertBatch
from loader.task.utils.base_classifiers import BertClusterClassifier, BertClassifier

from tinybert import BertOutput


class CurriculumClusterMLMTask(BaseCurriculumMLMTask, BaseClusterMLMTask):
    name = 'cu-cluster-mlm'
    cls_module = BertClassifier
    cluster_cls_module = BertClusterClassifier
    batcher = CurriculumMLMBertBatch

    def _rebuild_batch(self, batch):
        self.prepare_batch(batch)

        if self.is_training:
            self.random_mask(batch, self.k_cluster)
        self.left2right_mask(batch, self.p_cluster)

        return batch

    def produce_output(self, model_output, batch: CurriculumMLMBertBatch,test):
        # print('model_output[3]',model_output.shape)
        return self._produce_output(model_output, batch,test)

    def test__curriculum(self, batch: CurriculumMLMBertBatch, output, metric_pool,local_global_maps):
        return BaseClusterMLMTask.test__curriculum(
            self,
            batch,
            output,
            metric_pool=metric_pool,
            local_global_maps = local_global_maps
        )

    def calculate_loss(self, batch: CurriculumMLMBertBatch, output, **kwargs):
        return BaseClusterMLMTask.calculate_loss(
            self,
            batch=batch,
            output=output,
            weight=batch.weight,
            **kwargs
        )

    def test__left2right(self, samples, model, metric_pool, dictifier, k):
        for sample in samples:
            ground_truth = sample[self.p_global][:]
            length = len(ground_truth)
            arg_sorts = []

            times = (length + k - 1) // k
            print(sample)
            exit(0)
            for i in range(times):
                sample[self.p_global] = ground_truth[i * k: max((i + 1) * k, length)]
                batch = dictifier([self.dataset.build_format_data(sample)])
                batch = self.rebuild_batch(batch)  # type: CurriculumMLMBertBatch

                outputs = model(
                    batch=batch,
                    task=self,
                )[self.depot.get_vocab(self.concat_col)]  # [B, S, V]

                pred_cluster_labels = outputs['pred_cluster_labels'][0]
                outputs = outputs[self.p_local]
                col_mask = batch.mask_labels_col[self.p_cluster][0]
                cluster_indexes = [0] * self.n_clusters

                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[i_tok]:
                        cluster_id = pred_cluster_labels[i_tok]
                        top_items = mindspore.ops.argsort(
                            outputs[cluster_id][cluster_indexes[cluster_id]], descending=True
                        ).cpu().tolist()[:metric_pool.max_n]
                        top_items = [self.local_global_maps[cluster_id][item] for item in top_items]
                        arg_sorts.append(top_items)
                        cluster_indexes[cluster_id] += 1
                    else:
                        arg_sorts.append(None)

                # sample[self.k_]