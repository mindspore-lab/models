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
import json
import numpy as np

from sklearn import metrics
from longling import wf_open
from tqdm import tqdm

import mindspore
from mindspore.dataset import GeneratorDataset
from mindspore import context
from mindspore import nn

from EduSim.deep_model import DKTnet
from EduSim.utils import get_proj_path, mds_concat

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('GEHRL_mindspore')] + 'GEHRL_mindspore')  # 这里要改为你自己的项目的主目录


class MyDataset:
    def __init__(self, data_path, num_skills, feature_dim, max_sequence_length):
        self.data_path = data_path
        self.feature_dim = feature_dim
        self.num_skills = num_skills
        self.max_sequence_length = max_sequence_length
        self.datatxt = []
        with open(self.data_path, 'r', encoding="utf-8") as f:
            self.datatxt = f.readlines()
        print(f'Total training {len(self.datatxt)} sessions ')
        # self.get_selected_data()

    # 获取单条数据
    def __getitem__(self, index):
        # assert 0
        line = self.datatxt[index]
        # 一行是一个二维list，每个元素是[exer_id, 0 or 1] exer_id(0,834)
        session = json.loads(line)
        session = self.get_feature_matrix(session)
        return session  # [max_sequence_length, feature_dim]

    # 数据集长度
    def __len__(self):
        return len(self.datatxt)

    def get_feature_matrix(self, session):
        input_data = np.zeros(shape=(self.max_sequence_length, self.feature_dim), dtype=np.float32)
        # input_data = mindspore.ops.zeros(size=(self.max_sequence_length, self.feature_dim), dtype=mindspore.float32)
        # 对输入x进行编码
        j = 0
        while j < self.max_sequence_length and j < len(session):
            problem_id = session[j][0]
            if session[j][1] == 0:  # 对应问题回答错误
                input_data[j][problem_id] = 1.0
            elif session[j][1] == 1:  # 对应问题回答正确
                input_data[j][problem_id + self.num_skills] = 1.0
            j += 1
        return mindspore.Tensor(input_data)

    def get_data_correct_rate(self):
        # 统计学生回答正确在数据集中的比例
        answer_cor_num = 0
        all_num = 0
        for _, line in enumerate(self.datatxt):
            # 一行是一个二维list，每个元素是[exer_id, 0 or 1] exer_id(0,834)
            session = json.loads(line)
            data = [log[1] for log in session]
            all_num += len(data)
            answer_cor_num += np.array(data).sum().item()
        print(f'1 in data rate:{answer_cor_num / all_num}')

    def get_selected_data(self):
        # 精选一些比较好的做题记录用于训练一个可能更好的DKT模拟环境
        if self.data_path != f'{get_proj_path()}/data/dataProcess/junyi/student_log_kt_None':
            raise ValueError('This function can only be used under env dkt training situation')

        env_selected_train_data_path = f'{get_proj_path()}/data/dataProcess/junyi/student_log_kt_None_selected'
        selecte_count = 0
        with wf_open(env_selected_train_data_path) as wf:
            for _, line in tqdm(enumerate(self.datatxt), 'writing in file'):
                # 一行是一个二维list，每个元素是[exer_id, 0 or 1] exer_id(0,834)
                session = json.loads(line)
                different_questions = {log[0] for log in session}
                if len(different_questions) > 15:
                    selecte_count += 1
                    print(json.dumps(session), file=wf)
        print(f'final selected count:{selecte_count}')
        print(f'final selected rate:{selecte_count / len(self.datatxt)}')


class EnvDKTtrainer:
    def __init__(self, train_goal='env_DKT'):
        # 实验设置
        with open(f'{get_proj_path()}/data/dataProcess/junyi/graph_vertex.json') as f:
            # exercise-id的对应字典，存储 {exercise_str:id},length即为835
            ku_dict = json.load(f)
        self.num_skills = len(ku_dict)
        self.feature_dim = 2 * self.num_skills
        # 实验设置

        self.train_goal = train_goal
        self.test_only = False  # test acc: 0.8177, test auc: 0.8898
        self.env_selected_data_train = False
        print('Current training goal is:' + self.train_goal)


        # 训练参数
        if self.env_selected_data_train:
            self.env_dkt_data_path = f'{get_proj_path()}/data/dataProcess/junyi/student_log_kt_None_selected'
        else:
            self.env_dkt_data_path = f'{get_proj_path()}/data/dataProcess/junyi/student_log_kt_None'
        self.agent_dkt_data_path = f'{get_proj_path()}/data/dataProcess/junyi/dataOff'
        self.max_sequence_length = 200
        self.epoch_num = 5
        self.test_acc = 0.0
        self.test_auc = 0.0
        self.val_max_auc = 0.0
        self.test_max_auc = 0.0
        if self.train_goal == 'env_DKT':
            self.data_path = self.env_dkt_data_path
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KES/meta_data/env_weights/'
            os.makedirs(self.weight_path, exist_ok=True)
        else:
            self.data_path = self.agent_dkt_data_path
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KES/meta_data/agent_weights/'
            os.makedirs(self.weight_path, exist_ok=True)

        # 超参数
        self.learning_rate = 0.01
        self.embed_dim = 600  # from paper
        self.hidden_size = 900  # from paper
        self.batch_size = 256

        # 数据集
        self.dataset = MyDataset(self.data_path, self.num_skills, self.feature_dim, self.max_sequence_length)
        self.dataset = GeneratorDataset(source=self.dataset, column_names=['session'])
        self.dataset = self.dataset.batch(self.batch_size)

        (self.train_dataset, self.val_dataset,
         self.test_dataset) = self.dataset.split([0.8, 0.1, 0.1], randomize=False)
        self.train_size = self.train_dataset.get_dataset_size()
        self.val_size = self.val_dataset.get_dataset_size()
        self.test_size = self.test_dataset.get_dataset_size()

        # 模型
        dkt_input_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.01,
        }
        self.DKTnet = DKTnet(dkt_input_dict)

        # 优化器设置
        self.optimizer = nn.Adam(self.DKTnet.trainable_params(), learning_rate=self.learning_rate)
        self.loss_f = nn.BCEWithLogitsLoss()
        self.grad_fn = mindspore.value_and_grad(self.forward_fn,
                                                None, weights=self.DKTnet.trainable_params(), has_aux=True)

        # test or case study of a model
        # env_DKT如果设值threshold为0.5的话，ac为86%，如果设置为0.9则ac为75%
        if self.test_only:
            param_dict = mindspore.load_checkpoint(f'{self.weight_path}ValBest.ckpt')
            _, _ = mindspore.load_param_into_net(self.DKTnet, param_dict)
            self.test(name='Test', dataloader=self.test_dataset)
            assert 0

    def forward_fn(self, batch_data):
        output = self.DKTnet(batch_data)
        loss, batch_pred_probs, batch_true_labels, sequence_lengths = self.compute_loss(output, batch_data)
        return loss, batch_pred_probs, batch_true_labels, sequence_lengths

    def train(self):
        # 开始训练
        train_loss_list = []
        correct = 0
        true_labels = []
        pred_probs = []
        self.DKTnet.set_train()
        for epoch in range(self.epoch_num):
            for i, batch_data in enumerate(self.train_dataset.create_tuple_iterator()):
                (loss, batch_pred_probs, batch_true_labels, sequence_lengths), grads = self.grad_fn(*batch_data)
                self.optimizer(grads)

                train_loss_list.append(loss.asnumpy())
                if i % 20 == 0:
                    print(f"epcoh:{epoch + 1}  iteration:{i}   loss:{np.mean(train_loss_list):.6f}")

                batch_pred_labels = (batch_pred_probs >= 0.5).astype(mindspore.float32)
                for j in range(batch_pred_labels.shape[0]):
                    if sequence_lengths[j] == 1:
                        continue
                    a = batch_pred_labels[j, 0:sequence_lengths[j] - 1]
                    b = batch_true_labels[j, 1:sequence_lengths[j]]

                    correct = correct + mindspore.ops.equal(a, b).sum().asnumpy()
                    for p in batch_pred_probs[j, 0:sequence_lengths[j] - 1].view(-1):
                        pred_probs.append(p.asnumpy())
                    for t in batch_true_labels[j, 1:sequence_lengths[j]].view(-1):
                        true_labels.append(t.asnumpy())

            print(f'Cal train acc ..')
            acc = correct / len(true_labels)
            auc = metrics.roc_auc_score(true_labels, pred_probs)
            print(f'train acc:{acc}')
            print(f'train auc:{auc}')
            self.test('Val', self.val_dataset)

        self.test('test', self.test_dataset)

    def test(self, name, dataloader):
        print(f'...testing on {name} ...')
        test_loss = 0
        correct = 0
        true_labels = []
        pred_probs = []
        self.DKTnet.set_train(False)
        k = 0
        for i, batch_data in enumerate(dataloader.create_tuple_iterator()):
            y_hat = self.DKTnet(*batch_data)
            batch_loss, batch_pred_outs, batch_true_labels, sequence_lengths = self.compute_loss(y_hat, *batch_data)
            test_loss += batch_loss.asnumpy()
            batch_pred_labels = (batch_pred_outs >= 0.5).astype(mindspore.float32)

            for j in range(batch_pred_labels.shape[0]):
                if sequence_lengths[j] == 1:
                    continue
                a = batch_pred_labels[j, 0:sequence_lengths[j] - 1]
                b = batch_true_labels[j, 1:sequence_lengths[j]]

                correct = correct + mindspore.ops.equal(a, b).sum().asnumpy()
                for p in batch_pred_outs[j, 0:sequence_lengths[j] - 1].view(-1):
                    pred_probs.append(p.asnumpy())
                for t in batch_true_labels[j, 1:sequence_lengths[j]].view(-1):
                    true_labels.append(t.asnumpy())
            k = i

        test_loss /= (k + 1)

        acc = correct / len(pred_probs)
        auc = metrics.roc_auc_score(true_labels, pred_probs)
        print(f'Test result: Average loss: {float(test_loss):.4f}  '
              f'acc: {correct}/{len(pred_probs)}={acc}  '
              f'auc: {auc}')
        self.DKTnet.set_train()

        if name == 'Val':
            if self.val_max_auc < auc:
                self.val_max_auc = auc
                if 'env' in self.train_goal and self.env_selected_data_train:
                    mindspore.save_checkpoint(self.DKTnet, f'{self.weight_path}SelectedValBest.ckpt')
                else:
                    mindspore.save_checkpoint(self.DKTnet, f'{self.weight_path}ValBest.ckpt')
        else:
            self.test_acc = acc
            self.test_auc = auc

    def compute_loss(self, output, batch_data):
        sequence_lengths = [int(mindspore.ops.sum(sample)) for sample in batch_data]
        target_corrects = mindspore.Tensor([], dtype=mindspore.float32)
        target_ids = mindspore.Tensor([], dtype=mindspore.int32)
        output = output.permute(1, 0, 2)
        for episode in range(batch_data.shape[0]):
            tmp_target_id = mindspore.ops.Argmax(axis=-1)(batch_data[episode, :, :])
            ones = mindspore.ops.ones(tmp_target_id.shape, dtype=mindspore.float32)
            zeros = mindspore.ops.zeros(tmp_target_id.shape, dtype=mindspore.float32)
            # [sequence_length, 1]
            target_correct = mindspore.ops.where(
                tmp_target_id > self.num_skills - 1, ones, zeros).unsqueeze(1).unsqueeze(0)
            target_id = mindspore.ops.where(tmp_target_id > self.num_skills - 1,
                                            tmp_target_id - self.num_skills, tmp_target_id)  # [sequence_length]
            # target_id注意需要整体位移一格
            target_id = mindspore.ops.roll(target_id, -1, 0).unsqueeze(1).unsqueeze(0)  # [1, sequence_length, 1]
            # 放入batch里面
            target_ids = mds_concat((target_ids, target_id), 0)
            target_corrects = mds_concat((target_corrects, target_correct), 0)

        logits = output.gather_elements(dim=2, index=target_ids)
        preds = mindspore.ops.sigmoid(logits)
        loss = mindspore.Tensor([0.0])
        for i, sequence_length in enumerate(sequence_lengths):
            if sequence_length <= 1:
                continue
            a = logits[i, 0:sequence_length - 1]
            b = target_corrects[i, 1:sequence_length]
            loss = loss + self.loss_f(a, b)

        return loss, preds, target_corrects, sequence_lengths


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    context.set_context(device_target='GPU')
    handler = EnvDKTtrainer(train_goal='env_DKT')
    handler.train()
    handler_A = EnvDKTtrainer(train_goal='agent_DKT')
    handler_A.train()
