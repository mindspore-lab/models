# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train dcn"""

import mindspore as ms
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.communication.management import init
from src.deep_and_cross import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, DeepCrossModel
from src.callbacks import LossCallBack
from src.datasets import create_dataset, DataType
from src.config import DeepCrossConfig


def get_DCN_net(configure):
    """
    Get network of deep&cross model.
    """
    DCN_net = DeepCrossModel(configure)
    loss_net = NetWithLossClass(DCN_net)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(DCN_net)
    return train_net, eval_net

class StopAtStep(ms.Callback):
    """
    Start profiling base on step.

    Args:
        start_step (int): The start step number.
        stop_step (int): The stop step number.
    """
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = ms.Profiler(start_profile=False, output_path='./profiler_data2')

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
            self.profiler.analyse()
class ModelBuilder():
    """
    Build the model.
    """
    def __init__(self):
        pass

    def get_hook(self):
        pass

    def get_net(self, configure):
        return get_DCN_net(configure)


def test_train(configure):
    """
    test_train
    """
    if configure.is_distributed:
        if configure.device_target == "Ascend":
            context.set_context(device_id=configure.device_id, device_target=configure.device_target)
            init()
        elif configure.device_target == "GPU":
            context.set_context(device_target=configure.device_target)
            init("nccl")
        # context.set_context(mode=context.GRAPH_MODE)
        context.set_context(mode=context.PYNATIVE_MODE)
    else:
        context.set_context(mode=context.GRAPH_MODE,device_target=configure.device_target, device_id=configure.device_id)
        # context.set_context(mode=context.PYNATIVE_MODE, device_target=configure.device_target,device_id=configure.device_id)

    #profiler = ms.Profiler(output_path='./profiler_data')

    # profile_call_back = StopAtStep(1, 2)

    data_path = configure.data_path
    batch_size = configure.batch_size
    field_size = configure.field_size
    target_column = field_size + 1
    print("field_size: {} ".format(field_size))
    epochs = configure.epochs
    if configure.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif configure.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds_train = create_dataset(data_path, train_mode=True, epochs=1,
                              batch_size=batch_size, data_type=dataset_type, target_column=target_column)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    net_builder = ModelBuilder()
    train_net, _ = net_builder.get_net(configure)
    train_net.set_train()
    model = Model(train_net)
    callback = LossCallBack(config=configure)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(),
                                  keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='deepcross_train', directory=configure.ckpt_path, config=ckptconfig)
    model.train(epochs, ds_train, callbacks=[TimeMonitor(ds_train.get_dataset_size()), callback, ckpoint_cb],dataset_sink_mode=False)
    # model.train(epochs, ds_train, callbacks=[TimeMonitor(ds_train.get_dataset_size()), callback, ckpoint_cb,profile_call_back],dataset_sink_mode=False)

if __name__ == "__main__":
    config = DeepCrossConfig()
    config.argparse_init()
    test_train(config)
