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
import numpy as np
import mindspore
from mindspore import nn, ops


def get_raw_data_path():
    return 'xxx'  # change to your raw data path


def get_proj_path(proj_name='PKSD_mindspore'):
    """
    :param item_name: 项目名称，如pythonProject
    :return:
    """
    # 获取当前所在文件的路径
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取根目录
    return cur_path[:cur_path.find(proj_name)] + proj_name


def mean_entropy_cal(dist):  # [bz, cat_num]
    return mindspore.ops.mean(mindspore.ops.sum(dist * mindspore.ops.log(dist), dim=1))


def batch_cat_targets(DKT_states, targets, num_skills):
    target_tensor = mindspore.ops.zeros((DKT_states.shape[0], num_skills), dtype=mindspore.float32)

    for i, sample in enumerate(targets):
        index = mindspore.Tensor(sample, dtype=mindspore.int32)
        updates = mindspore.ops.ones(shape=index.shape)
        target_tensor[i] = mindspore.ops.tensor_scatter_elements(axis=0, indices=index,
                                                                 updates=updates, input_x=target_tensor[i])

    return mindspore.ops.cat((DKT_states, target_tensor), 1)  # [batch_size, 2*action_dim]


def get_feature_matrix(sequence, action_dim, embedding_dim, max_sequence_length=20):
    input_data = mindspore.ops.zeros((max_sequence_length, embedding_dim))

    # one_hot:
    # logs_encoding
    if sequence:
        index = mindspore.Tensor([int(item[0]) if item[1] == 0 else int(item[0]) + action_dim for item in sequence],
                                 dtype=mindspore.int32)
        input_data[:len(sequence)] = ops.one_hot(depth=embedding_dim, indices=index, axis=-1,
                                                 on_value=mindspore.Tensor(1.0, mindspore.float32),
                                                 off_value=mindspore.Tensor(0.0, mindspore.float32))

    return input_data


def episode_reward_reshape(episode_log, episode_reward):
    # reward reshaping
    items = [log[0] for i, log in enumerate(episode_log[-1][3])]
    for i, el in enumerate(episode_log):
        # during learning session, reward is 0.0
        el[2] = 0.0

        # last step's reward is episode reward
        if i == len(episode_log) - 1:
            el[2] = episode_reward
            # item diversity reward
            if episode_reward <= 0:
                el[2] = -1.0 / (len(list(set(items))))


def sample_reward_reshape(score, next_score, targets):
    reward = 0
    for i in targets:
        if score[i] == 0 and next_score[i] == 1:
            reward += 1
    return reward


def compute_dkt_loss(output, batch_data):
    loss_f = nn.BCEWithLogitsLoss()
    num_skills = int(batch_data[0].shape[1] / 2)
    sequence_lengths = [int(mindspore.ops.sum(sample)) for sample in batch_data]
    target_corrects = mindspore.Tensor([])
    target_ids = mindspore.Tensor([])
    output = output.permute(1, 0, 2)
    for episode in range(batch_data.shape[0]):
        tmp_target_id = mindspore.ops.Argmax(axis=-1)(batch_data[episode, :, :])
        ones = mindspore.ops.ones(tmp_target_id.shape, dtype=mindspore.float32)
        zeros = mindspore.ops.zeros(tmp_target_id.shape, dtype=mindspore.float32)
        # [sequence_length, 1]
        target_correct = mindspore.ops.where(tmp_target_id > num_skills - 1, ones, zeros).unsqueeze(1).unsqueeze(0)
        target_id = mindspore.ops.where(tmp_target_id > num_skills - 1, tmp_target_id - num_skills, tmp_target_id)
        # target_id注意需要整体位移一格
        target_id = mindspore.ops.roll(target_id, -1, 0).unsqueeze(1).unsqueeze(0)
        # 放入batch里面
        target_ids = mds_concat((target_ids, target_id), 0)
        target_corrects = mds_concat((target_corrects, target_correct), 0)
    logits = output.gather_elements(dim=2, index=target_ids)
    loss = mindspore.Tensor([0.0])
    for i, sequence_length in enumerate(sequence_lengths):
        if sequence_length <= 1:
            continue
        a = logits[i, 0:sequence_length - 1]
        b = target_corrects[i, 1:sequence_length]
        loss = loss + loss_f(a, b)
    return loss


def get_graph_embeddings(env_name):
    # node2vec
    if env_name == 'KSS':
        data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/KSSGraphEmbedding.npy'
    elif env_name == 'KES_junyi':
        data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/junyiGraphEmbedding.npy'
    elif env_name == 'KES_ASSIST15':
        data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/ASSISTGraphEmbedding.npy'
    elif env_name == 'KES_ASSIST09':
        data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/ASSIST09GraphEmbedding.npy'
    else:
        raise ValueError('Wrong graph embedding data path')

    # TransE
    # if env_name == 'KSS':
    #     data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/TransEKSSGraphEmbedding.npy'
    # elif env_name == 'KES_junyi':
    #     data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/TransEjunyiGraphEmbedding.npy'
    # elif env_name == 'KES_ASSIST15':
    #     data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/TransEASSISTGraphEmbedding.npy'
    # else:
    #     raise ValueError('Wrong graph embedding data path')

    try:
        embeddings = np.load(data_path)
    except FileNotFoundError:
        embeddings = None
        print(f'no {data_path} yet')
    return embeddings


def mds_concat(tensor_list, axis):
    if tensor_list[0].shape[0] == 0:
        return_thing = tensor_list[1]
    else:
        return_thing = mindspore.ops.cat(tensor_list, axis)
    return return_thing
