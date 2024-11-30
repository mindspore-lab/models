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

"""train for CBLOSS"""
import os
import time
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.experimental import optim
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from models.Resnet import resnet34
from CBloss import CB_loss
from dataset import load_train_lt, load_test
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num
from eval import test_epoch



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
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target)
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
    num_classes = config.num_classes
    model = resnet34(num_classes=num_classes)
    pretrain_weight_path = os.path.join(current_dir, config.pretrained_model)
    pretrain_weight = ms.load_checkpoint(pretrain_weight_path)     
    ms.load_param_into_net(model,pretrain_weight)
    imb_ratio = config.imb_ratio
    batch_size = config.batch_size
    # Load datasets
    dataset_test = load_test(batch_size=batch_size)
    num_each_class, dataset_train = load_train_lt(imb_ratio=imb_ratio,batch_size=batch_size)
    beta = config.beta
    gamma = config.gamma
    LR = config.LR 
    epoch_num = config.epoch_num
    loss_type = config.loss_type
    print('imb_ratio:{},LR:{},epoch_num:{},batch_size:{},num_each_class:{},loss_type:{},beta:{},gamma:{}'.format(
        imb_ratio,LR,epoch_num,batch_size,num_each_class,loss_type,beta,gamma
    ))
    optimizer = optim.SGD(model.trainable_params(), lr=LR, momentum=0.9, weight_decay=5e-4)
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = CB_loss(labels=label, logits=logits,
                    samples_per_cls=num_each_class, num_of_classes=10,
                    loss_type=loss_type, beta=beta, gamma=gamma)
        return loss, logits
    # Get gradient function
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss
    iterator = dataset_train.create_dict_iterator()
    dataset_size = dataset_train.get_dataset_size()
    max_acc = 0
    max_epoch = 0
    time_one_step_avg_list = []
    time_global_begin = time.perf_counter()
    for epoch in range(0,epoch_num):
        model.set_train(True)
        time_one_step_list = []
        for batch_idx, data in enumerate(iterator):
            image = data['image']
            label = data['label']
            time_one_step_begin = time.perf_counter()
            loss = float(train_step(image, label).asnumpy())
            time_one_step_end = time.perf_counter()
            time_one_step = (time_one_step_end - time_one_step_begin)*1000
            time_one_step_list.append(time_one_step)
            if batch_idx % config.log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, dataset_size,
                    100. * batch_idx / dataset_size, loss))
        acc = test_epoch(model,dataset_test)
        if acc > max_acc:
            max_acc = acc
            max_epoch = epoch
            ms.save_checkpoint(model, os.path.join(checkpoint_path, config.ckpt_file))
        print('Max_acc:{} from:{}/{} '.format(max_acc,max_epoch,epoch))
        print('time_one_step_avg:{}ms'.format(np.mean(time_one_step_list)))
        time_one_step_avg_list.append(np.mean(time_one_step_list))
    time_global_end = time.perf_counter()
    print('Max_acc:{} from:{}/{} '.format(max_acc,max_epoch,epoch))
    print('time_one_step_avg:{}ms'.format(np.mean(time_one_step_avg_list)))
    print('time_global:{}ms'.format((time_global_end-time_global_begin)*1000))

if __name__ == '__main__':
    run_train()



