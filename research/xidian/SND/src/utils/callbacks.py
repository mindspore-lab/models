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
from PIL import Image
from src.utils.metric_logger import SegmentationMetric, SimEntMetric
from tqdm import tqdm

def colorize_mask(mask):
    # mask: numpy array of the mask
    # classes num 6 : palette = [255,0,0, 255,255,255, 0, 0, 255, 0, 255, 0,255,255,0,0,255,255]
    # classes num 5 :
    palette = [128, 64, 128,
               244, 35, 232,
               70, 70, 70,
               102, 102, 156,
               190, 153, 153,
               153, 153, 153,
               250, 170, 30,
               220, 220, 0,
               107, 142, 35,
               152, 251, 152,
               70, 130, 180,
               220, 20, 60,
               255, 0, 0,
               0, 0, 142,
               0, 0, 70,
               0, 60, 100,
               # 0, 80, 100,
               0, 0, 230,
               119, 11, 32]
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


class StepMonitor(Callback):
    def __init__(self, per_print_times, ):
        super(StepMonitor, self).__init__()
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.loss_step = dict(loss_G=[], loss_D1=[], loss_D2=[])
        self.time_step_sum = 0.
        self.pbar = None

    def convert_loss(self, loss):
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]
                raise None

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))
        return loss

    def on_train_epoch_begin(self, run_context):
        self.pbar = tqdm(desc='Train Log:\t', dynamic_ncols=True)

    def on_train_step_begin(self, run_context):
        self.time_step_start = time.time()

    def on_train_step_end(self, run_context):
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
            # print(output_string)
            self.pbar.set_description_str(output_string)
            self.pbar.update(self._per_print_times)
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
        self.label_list = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                           'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']
        self.best_SimEnt = -0.1
        self.best_Ent = 1e5
        self.miou_SimEnt_selected = -0.1
        self.miou_Ent_selected = -0.1

    def evaluation(self, save_mask=False, return_simEnt=False):
        dataset = self.eval_dataset
        metric = SegmentationMetric(self.config.num_classes)
        simEnt_metric = SimEntMetric()
        self.net.set_train(False)
        pbar = tqdm(enumerate(dataset.create_dict_iterator()), total=dataset.get_dataset_size(), dynamic_ncols=True)
        for idx, data in pbar:
            images = data['image']
            labels = data['label']
            names = data['name']
            pred1, pred2 = self.net.net_G(images)
            size = labels.shape[-2:]
            pred1 = ops.ResizeBilinear(size, )(pred1)
            pred2 = ops.ResizeBilinear(size, )(pred2)
            metric.update(pred2, labels)
            simEnt_metric.update(pred1, pred2)
            Acc, mIoU, IoUs = metric.get(return_category_iou=True)
            pred1_en, pred2_en, ent = simEnt_metric.get()
            if (idx + 1) % (dataset.get_dataset_size() // 4) == 0:
                print('[Eval Sample : {}/{}]'.format(idx + 1, dataset.get_dataset_size()))
                Acc, mIoU, IoUs = metric.get(return_category_iou=True)
                print("Acc/mIoU : {:.2f}%/{:.2f},\t IoUs : {}".format(Acc * 100, mIoU * 100, IoUs))
            pbar.set_postfix(**{'mIoU': mIoU * 100, 'Acc': Acc, 'simEnt1': pred1_en, 'simEnt2': pred2_en, 'Ent': ent})
            pbar.update()

            if save_mask:
                for i, name in enumerate(names.asnumpy()):
                    pred = pred2.asnumpy()[i].argmax(0)
                    pred_vis = colorize_mask(pred)
                    os.makedirs(os.path.join(self.save_path, 'mask'), exist_ok=True)
                    save_path = os.path.join(self.save_path, 'mask', name)
                    pred_vis.save(save_path)
        Acc, mIoU, IoUs = metric.get(return_category_iou=True)
        IoUs_list = [(idx, label_name, iou) for idx, (label_name, iou) in enumerate(zip(self.label_list, IoUs))]
        print("Acc/mIoU : {:.2f}%/{:.2f}\t IoUs :\n {}".format(Acc * 100, mIoU * 100, IoUs_list))
        pred1_en, pred2_en, ent = simEnt_metric.get()
        print("pred1_en={:.2f}\t pred2_en={:.2f}\t Ent:{:.2f}".format(pred1_en, pred2_en, ent))

        self.net.set_train(True)
        if return_simEnt:
            return mIoU * 100, pred1_en, pred2_en, ent
        return mIoU * 100

    # def on_eval_epoch_begin(self, run_context):
    #     miou = self.evaluation(save_mask=True)
    #     print("The iou is {}".format(miou))

    def on_train_epoch_begin(self, run_context):
        miou,_,SimEnt,Ent = self.evaluation(return_simEnt=True)
        if miou > self.best_iou:
            checkpoint_path = os.path.join(self.save_path, 'best_iou_{:.2f}.ckpt'.format(self.best_iou))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            self.best_iou = miou
            checkpoint_path = os.path.join(self.save_path, 'best_iou_{:.2f}.ckpt'.format(self.best_iou))
            mindspore.save_checkpoint(self.net, checkpoint_path)
        if SimEnt > self.best_SimEnt:
            checkpoint_path = os.path.join(self.save_path, 'best_SimEnt_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_SimEnt,self.miou_SimEnt_selected))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            self.best_SimEnt = SimEnt
            self.miou_SimEnt_selected = miou
            checkpoint_path = os.path.join(self.save_path, 'best_SimEnt_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_SimEnt,self.miou_SimEnt_selected))
            mindspore.save_checkpoint(self.net, checkpoint_path)
        if Ent < self.best_Ent:
            checkpoint_path = os.path.join(self.save_path, 'best_Ent_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_Ent, self.miou_Ent_selected))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            self.best_Ent = Ent
            self.miou_Ent_selected = miou
            checkpoint_path = os.path.join(self.save_path, 'best_Ent_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_Ent,self.miou_Ent_selected))
            mindspore.save_checkpoint(self.net, checkpoint_path)
        print("the best iou is {:.2f}".format(self.best_iou))
        print("the best SimEnt is {:.2f} \t mIou is {:.2f}".format(self.best_SimEnt,self.miou_SimEnt_selected))
        print("the best Ent is {:.2f} \t mIou is {:.2f}".format(self.best_Ent,self.miou_Ent_selected))

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()

        if cb_params.cur_step_num % self.save_pred_every == 0:
            miou, _, SimEnt, Ent = self.evaluation(return_simEnt=True)
            if miou > self.best_iou:
                checkpoint_path = os.path.join(self.save_path, 'best_iou_{:.2f}.ckpt'.format(self.best_iou))
                if os.path.isfile(checkpoint_path):
                    os.remove(checkpoint_path)
                self.best_iou = miou
                checkpoint_path = os.path.join(self.save_path, 'best_iou_{:.2f}.ckpt'.format(self.best_iou))
                mindspore.save_checkpoint(self.net, checkpoint_path)
            if SimEnt > self.best_SimEnt:
                checkpoint_path = os.path.join(self.save_path, 'best_SimEnt_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_SimEnt, self.miou_SimEnt_selected))
                if os.path.isfile(checkpoint_path):
                    os.remove(checkpoint_path)
                self.best_SimEnt = SimEnt
                self.miou_SimEnt_selected = miou
                checkpoint_path = os.path.join(self.save_path, 'best_SimEnt_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_SimEnt, self.miou_SimEnt_selected))
                mindspore.save_checkpoint(self.net, checkpoint_path)
            if Ent < self.best_Ent:
                checkpoint_path = os.path.join(self.save_path, 'best_Ent_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_Ent, self.miou_Ent_selected))
                if os.path.isfile(checkpoint_path):
                    os.remove(checkpoint_path)
                self.best_Ent = Ent
                self.miou_Ent_selected = miou
                checkpoint_path = os.path.join(self.save_path, 'best_Ent_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_Ent, self.miou_Ent_selected))
                mindspore.save_checkpoint(self.net, checkpoint_path)
            print("the best iou is {:.2f}".format(self.best_iou))
            print("the best SimEnt is {:.2f} \t mIou is {:.2f}".format(self.best_SimEnt, self.miou_SimEnt_selected))
            print("the best Ent is {:.2f} \t mIou is {:.2f}".format(self.best_Ent, self.miou_Ent_selected))

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()

        miou, _, SimEnt, Ent = self.evaluation(return_simEnt=True)
        if miou > self.best_iou:
            checkpoint_path = os.path.join(self.save_path, 'best_iou_{:.2f}.ckpt'.format(self.best_iou))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            self.best_iou = miou
            checkpoint_path = os.path.join(self.save_path, 'best_iou_{:.2f}.ckpt'.format(self.best_iou))
            mindspore.save_checkpoint(self.net, checkpoint_path)
        if SimEnt > self.best_SimEnt:
            checkpoint_path = os.path.join(self.save_path, 'best_SimEnt_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_SimEnt, self.miou_SimEnt_selected))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            self.best_SimEnt = SimEnt
            self.miou_SimEnt_selected = miou
            checkpoint_path = os.path.join(self.save_path, 'best_SimEnt_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_SimEnt, self.miou_SimEnt_selected))
            mindspore.save_checkpoint(self.net, checkpoint_path)
        if Ent < self.best_Ent:
            checkpoint_path = os.path.join(self.save_path, 'best_Ent_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_Ent, self.miou_Ent_selected))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)
            self.best_Ent = Ent
            self.miou_Ent_selected = miou
            checkpoint_path = os.path.join(self.save_path, 'best_Ent_{:.2f}_miou_{:.2f}.ckpt'.format(self.best_Ent, self.miou_Ent_selected))
            mindspore.save_checkpoint(self.net, checkpoint_path)
        print("the best iou is {:.2f}".format(self.best_iou))
        print("the best SimEnt is {:.2f} \t mIou is {:.2f}".format(self.best_SimEnt, self.miou_SimEnt_selected))
        print("the best Ent is {:.2f} \t mIou is {:.2f}".format(self.best_Ent, self.miou_Ent_selected))
