# Copyright 2024 Xidian University
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
from model.model import Classifier, DCD, Encoder
from core.step3 import train_step3
from core.step2 import train_step2
from core.step1 import train_step1
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.utils import init_random_seed
from model_utils.config import config
from model.utils import eval_generator
import model.dataloader as dataloader
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore import Tensor as Tensor
from mindspore import ops as ops
from mindspore import context
import mindspore as ms
import time
import os
os.environ['GLOG_v'] = '3'


ms.set_seed(1)


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
                        print("unzip percent: {}%".format(
                            int(i * 100 / data_num)), flush=True)
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
        zip_file_1 = os.path.join(
            config.data_path, config.modelarts_dataset_unzip_name + ".zip")
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

        print("Device: {}, Finish sync unzip data from {} to {}.".format(
            get_device_id(), zip_file_1, save_dir_1))

    config.ckpt_save_dir = os.path.join(
        config.output_path, config.ckpt_save_dir)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, config.model_root)
    folder = os.path.exists(checkpoint_path)
    if not folder:
        os.makedirs(checkpoint_path)
    cfg = config
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=cfg.device_target)
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
    filepath = config.model_root
    if not os.path.isdir(os.path.join(current_dir, filepath)):
        os.mkdir(os.path.join(current_dir, filepath))
    time_begin_global = time.perf_counter()
    init_random_seed(config.manual_seed)

    # load dataset
    time_dataload_begin = time.perf_counter()
    src_train_dataloader = dataloader.mnist_dataloader(
        batch_size=config.batch_size, train=True)
    src_test_dataloader = dataloader.mnist_dataloader(
        batch_size=config.batch_size, train=False)
    tgt_test_dataloader = dataloader.svhn_dataloader(
        config.batch_size, split='test')
    time_dataload_end = time.perf_counter()
    time_dataload = time_dataload_end - time_dataload_begin
    print('time of dataload:{:.3f}ms'.format(time_dataload * 1000))

    # load models
    time_model_begin = time.perf_counter()
    encoder, classifier = Encoder(), Classifier()
    discriminator = DCD(input_features=config.disc_feature)
    params_src = list(encoder.trainable_params()) + \
        list(classifier.trainable_params())
    params_dcd = list(discriminator.trainable_params())
    optimizer_src = ms.nn.Adam(params_src, learning_rate=config.src_lr)
    optimizer_dcd = ms.nn.Adam(params_dcd, learning_rate=config.dcd_lr_2)
    loss_fn = ms.nn.CrossEntropyLoss()
    time_model_end = time.perf_counter()
    time_dataload = time_model_end - time_model_begin
    print('time of prepare models:{:.3f}ms'.format(time_dataload * 1000))

    print("=== Pretrain encoder and classifier for step 1 ===")
    time_src_train_begin = time.perf_counter()
    encoder, classifier = train_step1(
        encoder, classifier, src_train_dataloader, loss_fn, optimizer_src)
    time_src_train_end = time.perf_counter()
    time_step1_train = time_src_train_end - time_src_train_begin
    print('time of pretrain encoder and classifier for step 1:{:.3f}ms'.format(
        time_step1_train * 1000))

    print("=== Evaluating encoder and classifier for source domain ===")
    time_src_test_begin = time.perf_counter()
    param_dict_e = ms.load_checkpoint(os.path.join(
        current_dir, config.model_root, config.src_encoder_checkpoint))
    param_dict_c = ms.load_checkpoint(os.path.join(
        current_dir, config.model_root, config.src_classifier_checkpoint))
    ms.load_param_into_net(encoder, param_dict_e)
    ms.load_param_into_net(classifier, param_dict_c)
    eval_generator(encoder, classifier, src_test_dataloader)
    time_src_test_end = time.perf_counter()
    time_step1_test = time_src_test_end - time_src_test_begin
    print('time of test encoder and classifier for step 1:{:.3f}ms'.format(
        time_step1_test * 1000))

    print("=== Train DCD for step 2 ===")
    time_tgt_step2_train_begin = time.perf_counter()
    train_step2(encoder, discriminator, loss_fn, optimizer_dcd)
    time_tgt_step2_train_end = time.perf_counter()
    time_tgt_step2_train = time_tgt_step2_train_end - time_tgt_step2_train_begin
    print('time of train DCD for step 2:{:.3f}ms'.format(
        time_tgt_step2_train * 1000))

    print("=== Train encoder, classifier and discriminator for step 3 ===")
    time_tgt_step3_train_begin = time.perf_counter()
    param_dict_d = ms.load_checkpoint(os.path.join(
        current_dir, config.model_root, config.tgt_discriminator_checkpoint))
    ms.load_param_into_net(discriminator, param_dict_d)
    train_step3(encoder, classifier, discriminator,
                tgt_test_dataloader, loss_fn)
    time_tgt_step3_train_end = time.perf_counter()
    time_tgt_step3_train = time_tgt_step3_train_end - time_tgt_step3_train_begin
    print('time of train encoder, classifier and discriminator for step 3:{:.3f}ms'.format(
        time_tgt_step3_train * 1000))

    print("=== Test model perfermance ===")
    time_test_start = time.perf_counter()
    param_dict_e = ms.load_checkpoint(os.path.join(
        current_dir, config.model_root, config.tgt_encoder_checkpoint))
    ms.load_param_into_net(encoder, param_dict_e)
    param_dict_c = ms.load_checkpoint(os.path.join(
        current_dir, config.model_root, config.tgt_classifier_checkpoint))
    ms.load_param_into_net(classifier, param_dict_c)

    acc = eval_generator(encoder, classifier, tgt_test_dataloader)
    print('val accuracy:', acc)
    time_test_end = time.perf_counter()
    time_test = time_test_end - time_test_start
    print('time of test model perfermance:{:.3f}ms'.format(time_test * 1000))


if __name__ == '__main__':
    time_s = time.perf_counter()
    run_train()
    time_e = time.perf_counter()
    time_all = time_e - time_s
    print('time of All process:{:.3f}ms'.format(time_all * 1000))
    
