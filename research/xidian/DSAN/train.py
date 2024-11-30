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

"""train for DSAN"""
import os
import time
import math
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.communication.management import init, get_rank
from lmmd import LMMD_loss
from mindspore import ops,context,nn
from mindspore.context import ParallelMode
import mindspore as ms
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num
from models.DSAN import DSAN 
from data_loader import load_data
from eval import test


class WithLossCell(nn.Cell):
    def __init__(self, net ,loss_lmmd):
        super(WithLossCell, self).__init__(auto_prefix=True)
        self.net = net
        self.loss_lmmd = loss_lmmd

    def construct(self, data_source, data_target, label_source, weight):
        label_source_pred, label_target_pred,source_feature, target_feature = self.net(
            data_source, data_target)
        loss_lmmd = self.loss_lmmd(source_feature, target_feature, label_source, ops.Softmax(axis=1)(label_target_pred), label_source_pred, weight)
        return loss_lmmd

class Net(nn.Cell):
    def __init__(self, TrainOneStepCell):
        super(Net, self).__init__(auto_prefix=True)
        self.TrainOneStepCell = TrainOneStepCell

    def construct(self, data_source, data_target, label_source, weight):
        loss_lmmd = self.TrainOneStepCell(data_source, data_target, label_source, weight)
        loss_lmmd = loss_lmmd.mean()
        return loss_lmmd

def train_epoch(epoch, model, dataloaders, optimizer, config, time_one_step_avg_list):
    source_loader, target_train_loader, _ = dataloaders
    data_zip = enumerate(zip(source_loader, target_train_loader))
    loss_lmmd = LMMD_loss()
    lambd = 2 / (1 + math.exp(-10 * (epoch) / config.nepoch)) - 1
    weight = config.weight * lambd
    loss_net = WithLossCell(model, loss_lmmd)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    net = Net(train_net)
    net.set_train()
    time_one_epoch_begin = time.perf_counter()
    for step, ((data_source, label_source), (data_target, _)) in data_zip:
        Loss_lmmd = net(data_source, data_target, label_source, weight)
        if (step+1) % config.log_interval == 0:
            print('Epoch:{},Step:{},Loss_lmmd:{:.4f}'.format(epoch,step,Loss_lmmd.asnumpy()))
    time_one_epoch_end = time.perf_counter()
    time_one_epoch = (time_one_epoch_end-time_one_epoch_begin)*1000
    time_one_step_avg = time_one_epoch/(step+1)
    print("Epoch:{} time_one_epoch:{:.3f}ms time_one_step_avg:{:.3f}ms"
          .format(epoch,                  
                  time_one_epoch,
                  time_one_step_avg))
    time_one_step_avg_list.append(time_one_step_avg) 

def modelarts_pre_process():
    '''modelarts pre process function.'''

    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.ckpt_save_dir = os.path.join(config.output_path, config.ckpt_save_dir)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, config.model_root)
    folder = os.path.exists(checkpoint_path)
    if not folder:
        os.makedirs(checkpoint_path)
    cfg = config
    context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg.device_target)
    device_num = get_device_num()
    if cfg.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
    elif cfg.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        device_id = 0
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            device_id = get_rank()
    # Model setting
    time_global_begin=time.perf_counter()
    time_one_step_avg_list = []
    time_test_list = []
    SEED = config.seed
    np.random.seed(SEED)
    dataloaders = load_data(os.path.join(current_dir, config.dataset_path), config.src,
                            config.tar, config.batch_size)
    model = DSAN(num_classes=config.nclass)
    if config.bottleneck:
        optimizer = nn.SGD([
            {'params': model.feature_layers.trainable_params()},
            {'params': model.bottle.trainable_params(), 'lr': config.lr[1]},
            {'params': model.cls_fc.trainable_params(), 'lr': config.lr[2]},
        ], learning_rate=config.lr[0], momentum=config.momentum, weight_decay=config.decay)
    else:
        optimizer = nn.SGD([
            {'params': model.feature_layers.trainable_params()},
            {'params': model.cls_fc.trainable_params(), 'lr': config.lr[1]},
        ], learning_rate=config.lr[0], momentum=config.momentum, weight_decay=config.decay)
    best_accuracy=0
    for epoch in range(1, config.nepoch + 1):
        train_epoch(epoch, model, dataloaders, optimizer, config, time_one_step_avg_list)
        time_test_begin=time.perf_counter()
        accuracy = test(model, dataloaders[-1])
        time_test_end=time.perf_counter()
        time_test=(time_test_end-time_test_begin)*1000
        time_test_list.append(time_test)
        print('time of test:{:.3f}ms'.format(time_test))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            ms.save_checkpoint(model, os.path.join(checkpoint_path, config.ckpt_file))
    print('max accuracy: {:2%}'.format(best_accuracy))
    time_global_end=time.perf_counter()
    time_global=(time_global_end-time_global_begin)*1000
    print('time of train all:{:.3f}ms'.format(time_global))
    print('time of train one step:{:.3f}ms'.format(np.mean(time_one_step_avg_list)))
    print('time of test avg:{:.3f}ms'.format(np.mean(time_test_list)))


if __name__ == '__main__':
    run_train()



