import os.path
import time
from typing import Dict, List, Tuple

import mindspore as ms
from mindspore.train import RunContext

from segment_anything.dataset.dataset import create_dataloader
from segment_anything.evaluate.evaluator import Evaluator
from segment_anything.evaluate.metrics import create_metric
from segment_anything.utils import logger
from segment_anything.utils.registry import CALLBACK_REGISTRY
from mindspore.train import Callback


def sec_to_dhms(sec, append_sec_digit=True)-> List:
    """
    Args:
        append_sec_digit: if true, the digit part of second is appended to the result list,
        otherwise , it is combined as a float number in second.
    """
    dhms = [0]*4
    dhms[2], dhms[3] = divmod(sec, 60)  # min, sec
    dhms[1], dhms[2] = divmod(dhms[2], 60)  # hour, min
    dhms[0], dhms[1] = divmod(dhms[1], 23)  # day, hour
    for i in range(3):
        dhms[i] = int(dhms[i])
    if append_sec_digit:
        sec = int(dhms[3])
        dhms.append(dhms[3] - sec)
        dhms[3] = sec
    return dhms

def create_callback(args: List):
    """
    function to create callback list. This will then be feed into mindspore.train.Model class
    """
    cb_list = []
    if args is None: return cb_list

    for cb in args:
        cb_list.append(CALLBACK_REGISTRY.instantiate(**cb))
    return cb_list


@CALLBACK_REGISTRY.registry_module()
class EvalWhileTrain(Callback):
    def __init__(self,
                 data_loader: Dict,
                 metric: List,
                 input_column: List[List[str]],
                 interval=1,
                 start_epoch=1,
                 isolated_epoch=None):
        self.evaluator = None
        self.data_loader = create_dataloader(data_loader)
        self.metric = create_metric(metric)
        self.input_column = input_column
        self.interval = interval
        self.start_epoch = start_epoch
        self.isolated_epoch = isolated_epoch

    def on_train_epoch_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        eval_flag = False
        cur_epoch_id = cb_params.cur_epoch_num
        if (cur_epoch_id >= self.start_epoch and cur_epoch_id % self.interval == 0) \
            or cur_epoch_id == self.isolated_epoch:
            eval_flag = True
        if not eval_flag:
            return

        network = cb_params.network.network.net  # model without loss function, cb_params.network is train_one_step_cell
        if self.evaluator is None:
            self.evaluator = Evaluator(network, self.data_loader, metric=self.metric, input_column=self.input_column)

        logger.info(f'evaluate at epoch {cur_epoch_id}, interval is {self.interval}')
        self.evaluator.eval()


@CALLBACK_REGISTRY.registry_module()
class TrainStatusLog(Callback):
    """
    Callback to record the status of training, mainly including loss and time performance information.
    """
    def __init__(self, interval=100, loss_item=()):
        self.log_interval = interval
        self.loss_item = loss_item
        self.step_start_time = 0.0
        self.train_start_time = 0.0
        self.accumulate_loss = ms.Tensor(0.0, dtype=ms.float32)
        self.dataset_size = 0
        self.epoch_num = 0

    def on_train_begin(self, run_context: RunContext):
        cb_params = run_context.original_args()
        self.train_start_time = time.time()
        self.dataset_size = cb_params.train_dataset.get_dataset_size()
        self.epoch_num = cb_params.epoch_num

    def on_train_end(self, run_context: RunContext):
        logger.info('Training Finished')

    def on_train_step_begin(self, run_context: RunContext):
        self.step_start_time = time.time()
        self.interval_start_time = time.time()

    def on_train_step_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_epoch = cb_params.cur_epoch_num
        cur_step = cb_params.cur_step_num
        # assert len(loss) == len(self.loss_item) + 1
        self.accumulate_loss += loss

        if cur_step % self.log_interval == 0:
            lr = cb_params.train_network.optimizer.learning_rate(cur_step)
            smooth_loss = self.accumulate_loss / self.log_interval

            step_cost = time.time() - self.step_start_time
            train_already_cost = sec_to_dhms(time.time() - self.train_start_time)
            train_left = sec_to_dhms((self.dataset_size * self.epoch_num - cur_step) * step_cost)
            # loss_str = ', '.join(f'{n}:{v}' for n, v in zip(self.loss_item, loss[1:]))
            step_time_str = f'{step_cost:.2f}s'
            train_already_cost_str = f'{train_already_cost[0]}d {train_already_cost[1]:02d}:' \
                                     f'{train_already_cost[2]:02d}:{train_already_cost[3]:02d}'
            train_time_left_str = f'{train_left[0]}d {train_left[1]:02d}:{train_left[2]:02d}:{train_left[3]:02d}'

            logger.info(', '.join([
                f'glb_step[{cur_step}/{self.dataset_size * self.epoch_num}]',
                f'loc_step[{cur_step % self.dataset_size}/{self.dataset_size}]',
                f'epoch[{cur_epoch}/{self.epoch_num}]',
                f'loss[{loss.asnumpy():.4f}]',
                f'smooth_loss[{smooth_loss.asnumpy():.4f}]',
                f'lr[{lr.asnumpy():.7f}]',
                f'step_time[{step_time_str}]',
                f'already_cost[{train_already_cost_str}]',
                f'train_left[{train_time_left_str}]',
            ]))
            # reset
            self.accumulate_loss = ms.Tensor(0.0, dtype=ms.float32)


@CALLBACK_REGISTRY.registry_module()
class SaveCkpt(Callback):
    """
    Callback to record the status of training, mainly including loss and time performance information.
    """
    def __init__(self, interval=1, work_root='./work_root', save_dir='', main_device=True):
        """
        Args:
            work_root (str): the directory that ckpt directory is rooted at.
            save_dir (str): the directory (relative to work_root) to save ckpt.
             Note to leave this arg default and set automatically by set_directory_and_log method in train.py
            main_device (str): whether the current device is the main device.
             Note to leave this arg default and set automatically by set_directory_and_log method in train.py
        """
        self.interval = interval
        self.work_root = work_root
        self.save_dir = save_dir
        self.main_device = main_device
        print('main_device', self.main_device)
        self.full_save_dir = os.path.join(work_root, save_dir)


    def on_train_epoch_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        total_epoch_num = cb_params.epoch_num
        if self.main_device and (cur_epoch % self.interval == 0 or cur_epoch == total_epoch_num):
            save_path = os.path.join(self.full_save_dir, f'sam_{cur_epoch:03d}.ckpt')
            logger.info(f'saving ckpt of epoch {cur_epoch} at {save_path}, interval is {self.interval}')
            # model without loss function, cb_params.network is train_one_step_cell
            network = cb_params.network.network.net
            ms.save_checkpoint(network, save_path)


@CALLBACK_REGISTRY.registry_module()
class Profiler(Callback):
    def __init__(self, start_step=1, end_step=2, exit_after_analyze=True, out_dir='./my_prof'):
        self.start_step = start_step
        self.end_step = end_step
        self.exit_after_analyze = exit_after_analyze
        self.profiler = ms.Profiler(start_profile=False, output_path=out_dir)

    def on_train_step_begin(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step == self.start_step:
            logger.info(f'start analyzing profiler in step range [{self.start_step}, {self.end_step}]')
            self.profiler.start()

    def on_train_step_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step == self.end_step:
            self.profiler.stop()
            self.profiler.analyse()
            logger.info(f'finish analyzing profiler in step range [{self.start_step}, {self.end_step}]')
            if self.exit_after_analyze:
                run_context.request_stop()
