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

"""Common utilities"""

import logging

from mindspore import (ModelCheckpoint, CheckpointConfig,
                       LossMonitor)

from .callbacks import (SummaryCallbackWithEval, BestCheckpointSavingCallback,
                        TrainTimeMonitor, EvalTimeMonitor)


def get_callbacks(arch, rank, train_data_size, val_data_size, ckpt_dir,
                  best_ckpt_dir, summary_dir, ckpt_save_every_step=0,
                  ckpt_save_every_sec=0, ckpt_keep_num=10, print_loss_every=1,
                  collect_freq=0, collect_tensor_freq=None,
                  collect_input_data=False, keep_default_action=False,
                  logging_level=logging.INFO,
                  logging_format='%(levelname)s: %(message)s'):
    """Get common callbacks."""
    logging.basicConfig(format=logging_format, level=logging_level)
    if collect_freq == 0:
        collect_freq = train_data_size
    if ckpt_save_every_step == 0 and ckpt_save_every_sec == 0:
        ckpt_save_every_step = train_data_size
    config_ck = CheckpointConfig(
        # To save every epoch use data.train_dataset.get_data_size(),
        save_checkpoint_steps=ckpt_save_every_step,
        save_checkpoint_seconds=ckpt_save_every_sec,
        keep_checkpoint_max=ckpt_keep_num,
        append_info=['epoch_num', 'step_num']
    )
    train_time_cb = TrainTimeMonitor(data_size=train_data_size)
    eval_time_cb = EvalTimeMonitor(data_size=val_data_size)

    best_ckpt_save_cb = BestCheckpointSavingCallback(
        best_ckpt_dir, prefix=arch
    )

    ckpoint_cb = ModelCheckpoint(
        prefix=f'{arch}_{rank}',
        directory=ckpt_dir,
        config=config_ck
    )
    loss_cb = LossMonitor(print_loss_every)

    specified = {
        'collect_metric': True,
        'collect_train_lineage': True,
        'collect_eval_lineage': True,
        # "histogram_regular": "^network.*weight.*",
        # "collect_graph": True,
        # "collect_dataset_graph": True,
        'collect_input_data': collect_input_data,
    }
    summary_collector_cb = SummaryCallbackWithEval(
        summary_dir=summary_dir,
        collect_specified_data=specified,
        collect_freq=collect_freq,
        keep_default_action=keep_default_action,
        collect_tensor_freq=collect_tensor_freq
    )
    return [
        train_time_cb,
        eval_time_cb,
        ckpoint_cb,
        loss_cb,
        best_ckpt_save_cb,
        summary_collector_cb
    ]
