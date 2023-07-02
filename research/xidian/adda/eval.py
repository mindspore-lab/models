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
"""evaluation for adda """
import os
import time
import mindspore as ms
from mindspore import context
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num
from utils import get_data_loader
from models.lenet import LeNetClassifier, LeNetEncoder
from core.adapt import eval_tgt


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


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if device_target == "Ascend":
        context.set_context(device_id=get_device_id())
    src_classifier = LeNetClassifier()
    src_classifier_dict = ms.load_checkpoint(
        os.path.join(current_dir, config.model_root, config.src_classifier_checkpoint))
    ms.load_param_into_net(src_classifier, src_classifier_dict)
    tgt_encoder = LeNetEncoder()
    tgt_encoder_dict = ms.load_checkpoint(os.path.join(current_dir, config.model_root, config.tgt_encoder_checkpoint))
    ms.load_param_into_net(tgt_encoder, tgt_encoder_dict)
    tgt_data_loader_eval = get_data_loader(config.tgt_dataset, train=False)
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)

if __name__ == '__main__':
    run_eval()
