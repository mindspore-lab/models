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
"""Callbacks."""

import time
import logging
from pathlib import Path
from operator import lt, gt

import mindspore as ms
from mindspore._checkparam import Validator
from mindspore import Callback, SummaryCollector, SummaryRecord, RunContext


class BestCheckpointSavingCallback(Callback):
    """Callback to save best model checkpoints during training."""

    def __init__(
            self,
            ckpt_dir,
            target_metric='acc',
            best_is_max=True,
            prefix='',
            buffer=5
    ):
        """
        Initialize ckpt saving callback.

        Parameters
        ----------
        ckpt_dir: str
            Directory to save checkpoints to.
        target_metric: str
            Name of the metric listed in `metrics` parameter of Model.
        best_is_max: bool
            Flag to choose is the higher or lower metric value is better.
            For example:
                - if `target_metric=loss` then `best_is_max` should be False
                - if `target_metric=acc` then `best_is_max` should be True
        prefix: str
            Prefix of saved checkpoint file.
        buffer: int
            Max number of saved checkpoints.
        """
        self.ckpt_dir = Path(ckpt_dir)
        self._make_dir()
        self.target_metric = target_metric
        self.best_is_max = best_is_max
        self.prefix = prefix
        if best_is_max:
            self.best_metric = float('-inf')
            self.compare = lt
        else:
            self.best_metric = float('inf')
            self.compare = gt

        self.current_ckpt = []
        self.buffer_size = buffer

    def _make_dir(self):
        """Create a checkpoint directory."""
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)
            logging.info('Directory created %s', self.ckpt_dir)
        else:
            logging.warning('Directory already exists: %s', self.ckpt_dir)

    def _save_checkpoint(self, network, epoch):
        """
        Save checkpoint.

        Parameters
        ----------
        network
            Network to save checkpoint for.
        """
        # TODO: May not work with model arts or distributed training.
        if not float('-inf') < self.best_metric < float('inf'):
            return
        ckpt_name = \
            f'epoch={epoch}_{self.target_metric}={self.best_metric:.3f}.ckpt'
        if self.prefix:
            ckpt_name = f'{self.prefix}_{ckpt_name}'
        ms.save_checkpoint(network, str(self.ckpt_dir / ckpt_name))
        self.current_ckpt.append(self.ckpt_dir / ckpt_name)
        if len(self.current_ckpt) > self.buffer_size:
            removed = self.current_ckpt[0]
            removed.unlink()
            del self.current_ckpt[0]

    def on_eval_end(self, run_context: RunContext):
        """
        Check and safe checkpoint if needed after evaluation complete.

        Parameters
        ----------
        run_context: RunContext

        """
        cb_params = run_context.original_args()
        metrics = {k: v for k, v in cb_params.eval_results.items()}
        if self.target_metric not in metrics:
            raise KeyError(
                f'Target metric {self.target_metric} is not in '
                'cb_params.metrics.'
            )
        # If the new metric is better the previous "best"
        if self.compare(self.best_metric, metrics[self.target_metric]):
            self.best_metric = metrics[self.target_metric]
            self._save_checkpoint(
                cb_params.network, epoch=cb_params.cur_epoch_num
            )


class SummaryCallbackWithEval(SummaryCollector):
    """
    Callback that can collect a common information like SummaryCollector.

    Additionally, this callback collects:
        - learning rate
        - validation loss
        - validation accuracy
    """

    def __init__(
            self,
            summary_dir,
            collect_freq=10,
            collect_specified_data=None,
            keep_default_action=True,
            custom_lineage_data=None,
            collect_tensor_freq=None,
            max_file_size=None,
            export_options=None
    ):
        super().__init__(
            summary_dir,
            collect_freq,
            collect_specified_data,
            keep_default_action,
            custom_lineage_data,
            collect_tensor_freq,
            max_file_size,
            export_options
        )
        self.entered_count = 0

    def on_train_epoch_end(self, run_context: RunContext):
        """
        Collect learning rate after train epoch.

        Parameters
        ----------
        run_context: RunContext
        """
        cb_params = run_context.original_args()
        optimizer = cb_params.get('optimizer')
        if optimizer is None:
            optimizer = getattr(cb_params.network, 'optimizer', None)
        if optimizer is None:
            logging.warning('There is no optimizer found!')
        else:
            global_step = optimizer.global_step
            lr = optimizer.learning_rate(global_step)
            self._record.add_value('scalar', f'Train/learning_rate',
                                   ms.Tensor(lr))
            self._record.record(cb_params.cur_epoch_num)
            super().on_train_epoch_end(run_context)

    def on_eval_end(self, run_context: RunContext):
        """
        Collect metrics after evaluation complete.

        Parameters
        ----------
        run_context: RunContext
        """
        cb_params = run_context.original_args()
        metrics = {k: v for k, v in cb_params.eval_results.items()}
        logging.debug(
            'Result metrics for epoch %d: %s', cb_params.cur_epoch_num,
            str({key: metrics[key] for key in sorted(metrics)})
        )

        for metric_name, value in metrics.items():
            self._record.add_value(
                'scalar', f'Metrics/{metric_name}', ms.Tensor(value)
            )
        self._record.record(cb_params.cur_epoch_num)
        self._record.flush()

    def __enter__(self):
        """
        Enter in context manager and control that SummaryRecord created once.
        """
        if self.entered_count == 0:
            self._record = SummaryRecord(log_dir=self._summary_dir,
                                         max_file_size=self._max_file_size,
                                         raise_exception=False,
                                         export_options=self._export_options)
            self._first_step, self._dataset_sink_mode = True, True
        self.entered_count += 1
        return self

    def __exit__(self, *err):
        """
        Exit from context manager and control SummaryRecord correct closing.
        """
        self.entered_count -= 1
        if self.entered_count == 0:
            super().__exit__(err)


class TrainTimeMonitor(Callback):
    """
    Monitor the time in train process.

    Args:
        data_size (int): How many steps are the intervals between logging
            information each time.
            if the program get `batch_num` during training, `data_size`
            will be set to `batch_num`, otherwise `data_size` will be used.
            Default: None

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, data_size=None):
        super().__init__()
        self.data_size = data_size
        self.epoch_time = time.time()

    def on_train_epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
                For more details, please refer to :class:`mindspore.RunContext`
        """
        self.epoch_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Log process cost time at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
                For more details, please refer to :class:`mindspore.RunContext`
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        mode = cb_params.get("mode", "")
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num
        Validator.check_positive_int(step_size)

        step_seconds = epoch_seconds / step_size
        logging.info('%s epoch time: %5.3f ms, per step time: %5.3f ms',
                     mode.title(), epoch_seconds, step_seconds)


class EvalTimeMonitor(Callback):
    """
    Monitor the time in eval process.

    Args:
        data_size (int): How many steps are the intervals between logging
            information each time.
            if the program get `batch_num` during training, `data_size`
            will be set to `batch_num`, otherwise `data_size` will be used.
            Default: None

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, data_size=None):
        super().__init__()
        self.data_size = data_size
        self.epoch_time = time.time()

    def on_eval_epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
                For more details, please refer to :class:`mindspore.RunContext`
        """
        self.epoch_time = time.time()

    def on_eval_epoch_end(self, run_context):
        """
        Log process cost time at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
                For more details, please refer to :class:`mindspore.RunContext`
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        mode = cb_params.get("mode", "")
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num
        Validator.check_positive_int(step_size)

        step_seconds = epoch_seconds / step_size
        logging.info('%s epoch time: %5.3f ms, per step time: %5.3f ms',
                     mode.title(), epoch_seconds, step_seconds)



class StopAtEpoch(Callback):
    def __init__(self, summary_dir, start_epoch, stop_epoch):
        super(StopAtEpoch, self).__init__()
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.profiler = ms.Profiler(output_path=summary_dir, start_profile=False)
    def on_train_epoch_begin(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.start_epoch:
            self.profiler.start()
    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num == self.stop_epoch:
            self.profiler.stop()
    def on_train_end(self, run_context):
        self.profiler.analyse()
