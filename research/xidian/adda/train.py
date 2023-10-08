# Copyright 2021 Xidian University
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
"""train for ADDA."""
import os
import time
from core.adapt import eval_tgt, train_tgt
from core.pretrain import eval_src, train_src
from models.discriminator import Discriminator
from models.lenet import LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_random_seed
import mindspore as ms
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


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
    filepath = config.model_root
    if not os.path.isdir(os.path.join(current_dir, filepath)):
        os.mkdir(os.path.join(current_dir, filepath))
    time_begin_global = time.perf_counter()
    init_random_seed(config.manual_seed)
    # load dataset
    time_dataload_begin = time.perf_counter()
    src_data_loader = get_data_loader(config.src_dataset)
    src_data_loader_eval = get_data_loader(config.src_dataset, train=False)
    tgt_data_loader = get_data_loader(config.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(config.tgt_dataset, train=False)
    time_dataload_end = time.perf_counter()
    time_dataload = time_dataload_end - time_dataload_begin
    print('time of dataload:{:.3f}ms'.format(time_dataload * 1000))
    # load models
    src_encoder = LeNetEncoder()
    src_classifier = LeNetClassifier()
    tgt_encoder = LeNetEncoder()
    critic = Discriminator(input_dims=config.d_input_dims,
                           hidden_dims=config.d_hidden_dims,
                           output_dims=config.d_output_dims)
    # train source model (pretrain)
    print("=== Training classifier for source domain ===")
    time_src_train_begin = time.perf_counter()
    src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader)
    time_src_train_end = time.perf_counter()
    src_encoder = LeNetEncoder()
    src_classifier = LeNetClassifier()
    src_encoder_dict = ms.load_checkpoint(os.path.join(current_dir, config.model_root, config.src_encoder_checkpoint))
    src_classifier_dict = ms.load_checkpoint(
        os.path.join(current_dir, config.model_root, config.src_classifier_checkpoint))
    # load params to network
    ms.load_param_into_net(src_encoder, src_encoder_dict)
    ms.load_param_into_net(src_classifier, src_classifier_dict)
    # eval source model
    print("=== Evaluating classifier for source domain ===")
    time_src_test_begin = time.perf_counter()
    eval_src(src_encoder, src_classifier, src_data_loader_eval)
    print("=== Evaluating classifier for target domain ===")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    time_src_test_end = time.perf_counter()
    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    # init weights of target encoder with those of source encoder
    ms.load_param_into_net(tgt_encoder, src_encoder.parameters_dict())
    # train target model (adapt)
    time_tgt_train_begin = time.perf_counter()
    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader)
    time_tgt_train_end = time.perf_counter()
    time_end_global = time.perf_counter()
    # compute and print time_src_test
    time_src_test = (time_src_test_end - time_src_test_begin) * 1000
    print("time_src_test:{:.3f}ms".format(time_src_test))
    # compute and print time_src_train
    time_src_train = (time_src_train_end - time_src_train_begin) * 1000
    print("time_src_train:{:.3f}ms".format(time_src_train))
    # compute and print time_tgt_train
    time_tgt_train = (time_tgt_train_end - time_tgt_train_begin) * 1000
    print("time_tgt_train:{:.3f}ms".format(time_tgt_train))
    # compute and print time_global
    time_global = (time_end_global - time_begin_global) * 1000
    print("time_global:{:.3f}ms".format(time_global))
    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)

if __name__ == '__main__':
    run_train()
