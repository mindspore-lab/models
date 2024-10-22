import os

from absl import logging

from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context, Model
from mindspore.ops import composite as C
from mindspore import Tensor, Parameter, ParameterTuple
import src.constants as rconst
from src.dataset import create_dataset
import mindspore as ms
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id
from mindspore.train.dataset_helper import DatasetHelper
from src.dataset import DATASET_TO_NUM_USERS_AND_ITEMS
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.nn as nn
from copy import deepcopy


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)
        self.network = network
        self.weights = ParameterTuple(self.network.trainable_params())
        # self.sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)

    def construct(self, *input):
        gout = self.grad(self.network, self.weights)(*input)
        return gout


class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)
        self.network = network
        self.weights = self.network.weights

    def construct(self, *input):
        gout = self.grad(self.network, self.weights)(*input)
        return gout


class IF:
    def __init__(self, ds_train, ds_eval, loss_net, sens=1.0):
        self.ds_train = ds_train
        self.ds_eval = ds_eval
        self.ds_train_helper = DatasetHelper(ds_train, dataset_sink_mode=False)
        self.ds_eval_helper = DatasetHelper(ds_eval, dataset_sink_mode=False)
        self.loss_net = loss_net
        self.sens = sens
        self.expand_dims = ops.ExpandDims()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        # self.sec_grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.weights = ParameterTuple(loss_net.trainable_params())
        self.damp = 0.1
        self.scale = 25
        self.n_user, self.n_item = DATASET_TO_NUM_USERS_AND_ITEMS['ml-1m']
        self.first_grad = Grad(self.loss_net)
        self.sec_grad = GradSec(self.first_grad)

    def get_influence_function(self):
        hs = {}
        users = self.ds_eval.children[0].source._pos_users.tolist()
        items = self.ds_eval.children[0].source._pos_items.tolist()
        for i in range(self.n_user):
            user = ms.Tensor([users[i]])
            item = ms.Tensor([items[i]])
            # len_element = len(next_element)
            # #loss = loss_net(*next_element)
            # user,item, label=next_element
            label = ms.Tensor([1])
            valid_pt_mask = ms.Tensor([0.0])
            user = self.expand_dims(user, 0)
            item = self.expand_dims(item, 0)
            label = self.expand_dims(label, 0)
            valid_pt_mask = self.expand_dims(valid_pt_mask, 0)
            hs[int(user[0, 0].asnumpy())] = self.get_hessian_estimate(
                user, item, label, valid_pt_mask
            )
            # sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)  #
        g = ms.numpy.zeros((self.n_user, self.n_item), ms.float32)
        for next_element in self.ds_train_helper:
            users, items, labels, valid_pt_masks = next_element
            for i in range(len(users)):
                user = users[i : i + 1]
                item = items[i : i + 1]
                label = labels[i : i + 1]
                valid_pt_mask = valid_pt_masks[i : i + 1]
                loss = self.loss_net(user, item, label, valid_pt_mask)
                sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)  #
                t_grad = self.grad(self.loss_net, self.weights)(
                    user, item, label, valid_pt_mask, sens
                )
                g[int(user[0, 0].asnumpy())][
                    int(item[0, 0].asnumpy())
                ] += sum(  # the implementation of origin repo is opposite to paper
                    [
                        ms.Tensor.sum(k * j) if k != None and j != None else 0
                        for k, j in zip(t_grad, hs[int(user[0, 0].asnumpy())])
                    ]
                )
        return g

    def get_hessian_estimate(
        self, user, item, label, valid_pt_mask, damp=0.1, scale=25, recursion_depth=1
    ):
        loss = self.loss_net(user, item, label, valid_pt_mask)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)  #
        v_grad = self.grad(self.loss_net, self.weights)(
            user, item, label, valid_pt_mask, sens
        )
        h_estimate = deepcopy(v_grad)
        train_dataset_helper = DatasetHelper(self.ds_train, dataset_sink_mode=False)
        count = 0
        for next_element in train_dataset_helper:
            batch_users, batch_items, labels, valid_pt_mask = next_element
            hv = self.hvp(
                batch_users, batch_items, labels, valid_pt_mask, sens, h_estimate
            )
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                if _h_e != None and _v != None and _hv != None
                else 0
                for _v, _h_e, _hv in zip(v_grad, h_estimate, hv)
            ]
            count += 1
            if count == recursion_depth:
                break

        return h_estimate

    def hvp(self, user, item, label, valid_pt_mask, sens, cur_estimate):
        if len(self.weights) != len(cur_estimate):
            raise (ValueError("weights and h_estimate must have the same length."))
        # grads = self.grad(self.loss_net, self.weights)(user, item, label, valid_pt_mask,sens)
        hessians = []
        # hessians = self.sec_grad(self.grad, self.weights)(user, item, label, valid_pt_mask, sens)
        # grad_2 = []
        # for g in grads:
        #     grad_2.append(self.grad(g, self.weights)(user, item, label, valid_pt_mask, sens))
        hessians = self.sec_grad(user, item, label, valid_pt_mask)
        return hessians
