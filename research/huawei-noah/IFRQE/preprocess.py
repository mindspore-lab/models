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
"""Using for eval the model checkpoint"""
import os

from absl import logging

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context, Model
from mindspore.ops import composite as C
from mindspore import Tensor, Parameter, ParameterTuple
import src.constants as rconst
from src.dataset import create_dataset, mask_dataset
from src.metrics import IFRQEMetric
from src.ncf import NCFModel, NetWithLossClass, TrainStepWrap, PredictWithSigmoid

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id
from mindspore.train.dataset_helper import DatasetHelper
from src.influence_function import IF
import mindspore as ms
import numpy as np
import math
from mindspore.nn import Softmax
from src.dataset import DATASET_TO_NUM_USERS_AND_ITEMS

logging.set_verbosity(logging.INFO)


def get_action_by_descision(n_user, n_item, decision, history, action_len):
    id = []
    o = ms.numpy.rand((n_user, n_item))
    for i in range(n_user):
        for k in range(action_len[i]):
            for j in range(len(history[i])):
                if (k >> j) & 1 == 1:
                    o[i, int(history[i][j])] = o[i, int(history[i][j])] + decision[i][k]
        o = o > 0.5
        mi = 1
        act = 0
        for j in range(len(history[i])):
            if o[i, j] == True:
                act += mi
            mi *= 2
        id.append(act)
    return o, id


@moxing_wrapper()
def run_eval():
    """eval method"""
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=config.device_target,
        save_graphs=False,
        device_id=get_device_id(),
    )

    layers = config.layers
    num_factors = config.num_factors
    topk = rconst.TOP_K
    num_eval_neg = rconst.NUM_EVAL_NEGATIVES

    ds_eval, num_eval_users, num_eval_items, _, _ = create_dataset(
        test_train=False,
        data_dir=config.data_path,
        dataset=config.dataset,
        train_epochs=0,
        eval_batch_size=config.eval_batch_size,
    )
    ds_train, num_eval_users, num_eval_items, history, history_len = create_dataset(
        test_train=True,
        data_dir=config.data_path,
        dataset=config.dataset,
        train_epochs=1,
        eval_batch_size=config.eval_batch_size,
    )
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    ncf_net = NCFModel(
        num_users=num_eval_users,
        num_items=num_eval_items,
        num_factors=num_factors,
        model_layers=layers,
        mf_regularization=0,
        mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
        mf_dim=16,
    )
    param_dict = load_checkpoint(config.checkpoint_file_path)
    load_param_into_net(ncf_net, param_dict)

    loss_net = NetWithLossClass(ncf_net)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(ncf_net, topk, num_eval_neg)
    lam = 1
    lr = 0.01
    L = 1
    inf_func = IF(ds_train, ds_eval, loss_net)
    g = inf_func.get_influence_function()
    n_user, n_item = DATASET_TO_NUM_USERS_AND_ITEMS['ml-1m']
    decision = ms.numpy.zeros((n_user, pow(2, 21)), ms.float32)
    dtemp = ms.numpy.zeros((n_user, pow(2, 21)), ms.float32)
    action_len = np.power(2, history_len)
    softmax = Softmax()
    unwill = ms.numpy.rand((n_user, n_item))
    actual = ms.numpy.zeros((n_user, n_item))
    for i in range(n_user):
        for j in range(history_len[i]):
            actual[i][int(history[i][j])] = 1
    unwill = unwill * actual
    for i in range(n_user):
        b = ms.numpy.rand(int(action_len[i]))
        decision[i][: int(action_len[i])] = softmax(b)
    for m in range(L):
        o, id = get_action_by_descision(n_user, n_item, decision, history, history_len)
        go = ms.Tensor.sum(g * o)
        un = lam * ms.Tensor.sum(o * unwill, axis=1)
        temp = ms.Tensor.sub(go, un)
        for i in range(n_user):
            gl = -temp[i]
            decision[i][id[i]] *= math.exp(-lr * gl)
            tot = ms.Tensor.sum(decision)
            decision[i] = decision[i] / tot
            dtemp[i] += decision[i]
    decision = dtemp / L
    final_action, _ = get_action_by_descision(
        n_user, n_item, decision, history, history_len
    )
    regret = ms.Tensor.sum(final_action * unwill)
    regret = regret / n_user
    tot_regret = ms.Tensor.sum(unwill)
    tot_regret = tot_regret / n_user

    mask_dataset(ds_train, final_action, config.data_path, config.dataset)
    # ncf_metric = NCFMetric()
    # model = Model(train_net, eval_network=eval_net, metrics={"ncf": ncf_metric})

    # ncf_metric.clear()
    # out = model.eval(ds_eval)

    # eval_file_path = os.path.join(config.output_path, config.eval_file_name)
    # eval_file = open(eval_file_path, "a+")
    # eval_file.write("EvalCallBack: HR = {}, NDCG = {}\n".format(out['ncf'][0], out['ncf'][1]))
    # eval_file.close()
    # print("EvalCallBack: HR = {}, NDCG = {}".format(out['ncf'][0], out['ncf'][1]))
    # print("=" * 100 + "Eval Finish!" + "=" * 100)


if __name__ == '__main__':
    run_eval()
