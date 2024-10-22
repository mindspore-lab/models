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


from oba import Obj
from loader.task.task_manager import TaskManager
import argparse
from model_bert import AutoBert
from utils.config_initializer import ConfigInitializer
from load_mindspore_csv import get_local_global_maps
from loader.init.bert_init import BertInit
from loader.data import EmbeddingInit, TaskInitializer
from tqdm import tqdm
import mindspore
from mindspore import nn
import utils.metric as metric
from load_mindspore_csv import Batch,get_dataset_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/aotm-bert-double-n10.yaml')
    parser.add_argument('--exp', type=str, default='exp/curriculum-bert-double-step.yaml')
    parser.add_argument('--cuda', type=int, default=None)
    parser.add_argument('--display_batch', type=int, default=0)
    args = parser.parse_args()
    config, exp = ConfigInitializer.init(args.config, args.exp)
    task_manager = TaskManager(exp)
    mode = 'train'
    tasks = task_manager.tasks
    print('tasks',tasks)
    embedding_init = EmbeddingInit()
    for embedding_info in config.embedding:

        embedding_init.append(**Obj.raw(embedding_info), global_freeze=exp.freeze_emb)
    model_init = BertInit(
        embedding_init=embedding_init,
        global_freeze=exp.freeze_emb,
        **Obj.raw(config.model_config),
    )
    task_initializer = TaskInitializer(
        model_init=model_init,
        device='CPU',
    ).register(*tasks)
    print('task_initializer',task_initializer)
    auto_model = AutoBert(
        device='CPU',
        model_init=model_init,
        task_initializer=task_initializer,
    )
    m_optimizer = nn.Adam(
        params=auto_model.trainable_params(),
        learning_rate=exp.policy.lr
    )
    def forward_fn(data):
        tasks[0].train()
        batch1 = Batch(data)
        task_output = auto_model(batch1, tasks[0])
        loss1 = tasks[0].calculate_loss(batch1, task_output, model=auto_model)
        return loss1, task_output['pred_cluster_labels']


    grad_fn = mindspore.value_and_grad(forward_fn, None, m_optimizer.parameters, has_aux=False)


    def train_step(data):
        (loss, _), grads = grad_fn(data)
        m_optimizer(grads)
        return loss


    def train_loop(model, dataset):
        size = dataset.get_dataset_size()
        model.set_train()
        for batch, data in enumerate(tqdm(dataset.create_dict_iterator())):
            loss = train_step(data)
            if batch % 50 == 0:
                loss, current = loss.asnumpy(), batch
                print(f"loss: {loss:>7f},  step: [{current} : {size}]")


    def test_loop():
        dataset = get_dataset_all('./data/ListContUni/aotm-n10/test_all.csv',batch_size=10)
        metric_pool = metric.MetricPool()
        local_global_maps = get_local_global_maps()
        metric_pool.add(metric.NDCG(), ns=[10])
        metric_pool.add(metric.HitRate(), ns=[10])
        metric_pool.init()
        for index, data in enumerate(tqdm(dataset.create_dict_iterator())):
            batch = Batch(data)
            task_output = auto_model(batch, tasks[0],test=True)
            predict = task_output
            tasks[0].test__curriculum(batch, predict, metric_pool,local_global_maps)
        metric_pool.export()
        for metric_name, n in metric_pool.values:
            if n:
                print(f'{metric_name}@{n:4d}: {metric_pool.values[(metric_name, n)]:.4f}')
            else:
                print(f'{metric_name}     : {metric_pool.values[(metric_name, n)]:.4f}')

    dataset = get_dataset_all('./data/ListContUni/aotm-n10/train_all.csv', batch_size=100)
    for i in range(10):
        train_loop(auto_model, dataset)
        test_loop()

