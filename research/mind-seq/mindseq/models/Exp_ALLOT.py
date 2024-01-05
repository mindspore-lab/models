import logging

logging.basicConfig(level=logging.INFO)
import random
import numpy as np
from ..data.dataset import TrafficDataset
from .layers.mode import Mode
from .layers.architecture import Architecture
from .ALLOT import ALLOT
from .run_manager import RunManager
from mindspore.communication import get_group_size, get_rank, init

from .Exp_basic import Exp_Basic

# import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_ALLOT(Exp_Basic):
    def __init__(self, args):
        super(Exp_ALLOT, self).__init__(args)

        self.run_manager = RunManager(
            path=self.args.param_path,
            name=self.model_name,
            net=Architecture(self.model),
            dataset=self.dataset,

            arch_lr=self.args.arch_lr,
            arch_lr_decay_milestones=self.args.arch_lr_decay_milestones,
            arch_lr_decay_ratio=self.args.arch_lr_decay_ratio,
            arch_decay=self.args.arch_decay,
            arch_clip_gradient=self.args.arch_clip_gradient,

            weight_lr=self.args.weight_lr,
            weight_lr_decay_milestones=self.args.weight_lr_decay_milestones,
            weight_lr_decay_ratio=self.args.weight_lr_decay_ratio,
            weight_decay=self.args.weight_decay,
            weight_clip_gradient=self.args.weight_clip_gradient,

            num_search_epochs=self.args.num_search_epochs, 
            num_train_epochs=self.args.num_train_epochs,

            criterion=self.args.criterion,
            metric_names=self.args.metric_names,
            metric_indexes=self.args.metric_indexes,
            print_frequency=self.args.print_frequency,

            use_gpu=self.args.use_gpu,
            device_ids=self.args.device_ids,
            
            reduce_flag=self.args.distribute,
        )
        self.run_manager._load(exp_mode=self.args.load_mode)
        self.run_manager.clear_records()
        self.run_manager.initialize()
        net_modes = {
            'one_fixed':Mode.ONE_PATH_FIXED,
            'one_random':Mode.ONE_PATH_RANDOM,
            'two_path':Mode.TWO_PATHS,
            'all_path':Mode.ALL_PATHS,
            'project':Mode.PROJECT
        }
        self.args.net_mode = net_modes[self.args.net_mode]

    def _build_model(self):
        self.model_name = self.args.name + '_' + self.args.desc
        # load data
        self.dataset = TrafficDataset(
            data_path=self.args.data_path,
            path=self.args.path,
            train_prop=self.args.train_prop,
            test_prop=self.args.test_prop,
            num_sensors=self.args.num_sensors,
            normalized_k=self.args.normalized_k,
            adj_type=self.args.adj_type,
            in_length=self.args.in_length,
            out_length=self.args.out_length,
            batch_size=self.args.batch_size,
            device_num=self.args.device_num,
            rank_id=self.args.rank_id,
        )
        model = ALLOT(
            adjinit=self.dataset.adj_mats[:,:,0],
            nodes=self.args.num_sensors,
            in_length=self.args.in_length,
            out_length=self.args.out_length,
            in_size=self.args.in_size,
            out_size=self.args.out_size,
            hidden_size=self.args.hidden_size,
            skip_size=self.args.skip_size,
            layer_names=self.args.layer_names,
            skip_mode=self.args.skip_mode,
            node_out=self.args.node_out,
            num_nodes=self.args.num_nodes,
            candidate_op_profiles=self.args.candidate_op_profiles,
            dropout=self.args.dropout
        )
        return model

    def train(self):
        self.run_manager.train(self.args.epoch, self.args.net_mode, 0 if self.args.rank_id is None else self.args.rank_id)

    def test(self):
        self.run_manager.test(self.args.net_mode)