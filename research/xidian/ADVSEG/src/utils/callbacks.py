# Copyright 2023 Xidian University
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

import mindspore

from src.model_utils import evaluation
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator
from mindspore.train.callback import Callback
from mindspore import ops
import time
import numpy as np


class StepMonitor(Callback):
    def __init__(self, per_print_times):
        super(StepMonitor, self).__init__()
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.loss_step = dict(loss_G=[], loss_D1=[], loss_D2=[])
        self.time_step_sum = 0.

    def convert_loss(self, loss):
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]
                raise None

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))
        return loss

    def step_begin(self, run_context):
        self.time_step_start = time.time()

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        loss_G, loss_D1, loss_D2 = loss
        for key in self.loss_step:
            self.loss_step[key].append(self.convert_loss(locals()[key]))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        # if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
        #     raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
        #         cb_params.cur_epoch_num, cur_step_in_epoch))

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        # if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
        #     while cb_params.cur_step_num <= self._last_print_time:
        #         self._last_print_time -= \
        #             max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        self.time_step_sum += time.time() - self.time_step_start
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            per_time = self.time_step_sum  # self._per_print_times
            # per_time = time.time() - self.time_step_start
            self.time_step_sum = 0.

            output_string = '[Info:' + time.strftime("%Y-%m-%d %H:%M:%S | ", time.localtime())
            output_string += '#Epoch:{}/{} step:{}/{} | #'.format(cb_params.cur_epoch_num, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num)
            for key, value in self.loss_step.items():
                output_string += 'avg_{:s}={:.4f} '.format(key, np.mean(value))
                self.loss_step[key] = []
            output_string += '| #per time:{:.3f}s ]'.format(per_time)
            print(output_string)
            # ops.Print()(output_string)
            # print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)


class CheckpointMonitor(Callback):
    def __init__(self, config, net, eval_dataset, ):
        super(CheckpointMonitor, self).__init__()
        self.net = net
        self.config = config
        self.save_pred_every = config.save_pred_every
        self.save_path = config.snapshot_dir
        self.eval_dataset = eval_dataset
        os.makedirs(self.save_path, exist_ok=True)
        self.best_iou = -0.1

    # def begin(self, run_context):
    #     config = self.config
    #     miou = evaluation(self.net.model_G, self.eval_dataset.create_dict_iterator(), ops.ResizeBilinear(size=(1024, 2048)),
    #                       config.data_dir_target,
    #                       config.save_result, config.data_list_target, logger=None, save=True, config=config)
    #     if miou > self.best_iou:
    #         checkpoint_path = os.path.join(self.save_path, 'best_{}.ckpt'.format(self.best_iou))
    #         if os.path.isfile(checkpoint_path):
    #             os.remove(checkpoint_path)
    #         self.best_iou = miou
    #         checkpoint_path = os.path.join(self.save_path, 'best_{}.ckpt'.format(self.best_iou))
    #         mindspore.save_checkpoint(self.net, checkpoint_path)
    #     print("the best iou is {}".format(self.best_iou))

    def step_end(self, run_context):
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % self.save_pred_every == 0:
            checkpoint_path = os.path.join(self.save_path, 'step_{}.ckpt'.format(cb_params.cur_step_num))
            mindspore.save_checkpoint(self.net, checkpoint_path)

            config = self.config
            miou = evaluation(self.net.model_G, self.eval_dataset.create_dict_iterator(), ops.ResizeBilinear(size=(1024, 2048)),
                              config.data_dir_target,
                              config.save_result, config.data_list_target, logger=None, save=True, config=config)
            if miou > self.best_iou:
                checkpoint_path = os.path.join(self.save_path, 'best_{}.ckpt'.format(self.best_iou))
                if os.path.isfile(checkpoint_path):
                    os.remove(checkpoint_path)
                self.best_iou = miou
                checkpoint_path = os.path.join(self.save_path, 'best_{}.ckpt'.format(self.best_iou))
                mindspore.save_checkpoint(self.net, checkpoint_path)
            print("the best iou is {}".format(self.best_iou))

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % self.save_pred_every == 0:
            checkpoint_path = os.path.join(self.save_path, 'step_{}.ckpt'.format(cb_params.cur_step_num))
            mindspore.save_checkpoint(cb_params.train_network, checkpoint_path)

        config = self.config
        miou = evaluation(self.net.model_G, self.eval_dataset.create_dict_iterator(), ops.ResizeBilinear(size=(1024, 2048)),
                          config.data_dir_target,
                          config.save_result, config.data_list_target, logger=None, save=True, config=config)
        if miou > self.best_iou:
            checkpoint_path = os.path.join(self.save_path, 'best_{}.ckpt'.format(self.best_iou))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            self.best_iou = miou
            checkpoint_path = os.path.join(self.save_path, 'best_{}.ckpt'.format(self.best_iou))
            mindspore.save_checkpoint(self.net, checkpoint_path)
        print("the best iou is {}".format(self.best_iou))
