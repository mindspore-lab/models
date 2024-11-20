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
from models.models import MyWithLossCell,TrainOneStepCellV2
import task_generator as tg
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

import os
import time
import random
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import ops
from mindspore.common.initializer import Normal, initializer
from model_utils.config import config


# 初始化环境
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target,device_id=config.DEVICE_ID) 
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


# 初始化数据和存储路径
(metatrain_character_folders,metatest_character_folders,) = tg.omniglot_character_folders()
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(current_dir, config.model_root)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
encoder_checkpoint = os.path.join(current_dir, config.model_root, config.encoder_checkpoint)
relation_checkpoint = os.path.join(current_dir, config.model_root, config.relation_checkpoint)


# 初始化网络

class CNNEncoder(nn.Cell):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer01 = nn.SequentialCell(
                        nn.Conv2d(1,64,kernel_size=3,pad_mode='valid'),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2,pad_mode='pad'))
        self.layer02 = nn.SequentialCell(
                        nn.Conv2d(64,64,kernel_size=3,pad_mode='valid'),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2,pad_mode='pad'))
        self.layer03 = nn.SequentialCell(
                        nn.Conv2d(64,64,kernel_size=3,pad_mode='pad', padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer04 = nn.SequentialCell(
                        nn.Conv2d(64,64,kernel_size=3,pad_mode='pad', padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        
        for layer in self.cells():
            for name, param in layer.parameters_and_names():
                if "weight" in name:
                    param.set_data(initializer(Normal(), param.shape, param.dtype))
                if "bias" in name:
                    param.set_data(initializer("zeros", param.shape, param.dtype))
                

    def construct(self,x):
        out = self.layer01(x)
        out = self.layer02(out)
        out = self.layer03(out)
        out = self.layer04(out)
        return out # 64

class RelationNetwork(nn.Cell):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.SequentialCell(
                        nn.Conv2d(128,64,kernel_size=3,pad_mode='pad', padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2,pad_mode='pad'))
        self.layer2 = nn.SequentialCell(
                        nn.Conv2d(64,64,kernel_size=3,pad_mode='pad', padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2,2,pad_mode='pad'))
        self.fc1 = nn.Dense(input_size,hidden_size)
        self.fc2 = nn.Dense(hidden_size,1)
        
        for layer in self.cells():
            for name, param in layer.parameters_and_names():
                if "weight" in name:
                    param.set_data(initializer(Normal(), param.shape, param.dtype))
                if "bias" in name:
                    param.set_data(initializer("zeros", param.shape, param.dtype))
                

    def construct(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0],-1)
        out = ops.relu(self.fc1(out))
        out = ops.sigmoid(self.fc2(out))
        return out


feature_encoder = CNNEncoder()
relation_network = RelationNetwork(64, 8)


milestone = [100000, 100000 * 2, 100000 * 3]
learning_rates = [config.LEARNING_RATE, config.LEARNING_RATE * 0.5, config.LEARNING_RATE * 0.5 * 0.5]
learning_rates = nn.piecewise_constant_lr(milestone, learning_rates)
params = list(feature_encoder.trainable_params()) + \
    list(relation_network.trainable_params())   
optimizer = nn.Adam(params,learning_rate=learning_rates)
loss_net = MyWithLossCell(Net1=feature_encoder,Net2=relation_network, loss_fn=nn.MSELoss())
train_net = TrainOneStepCellV2(loss_net, optimizer=optimizer)



def train_one_step(net,folder):
    net.set_train()
    degrees = random.choice([0, 90, 180, 270])
    task = tg.OmniglotTask(folder,config.CLASS_NUM,config.SAMPLE_NUM_PER_CLASS,config.BATCH_NUM_PER_CLASS)
    sample_dataloader = tg.get_data_loader(
        task,
        num_per_class=config.SAMPLE_NUM_PER_CLASS,
        split="train",
        shuffle=False,
        rotation=degrees,
    )
    batch_dataloader = tg.get_data_loader(
        task,
        num_per_class=config.BATCH_NUM_PER_CLASS,
        split="test",
        shuffle=True,
        rotation=degrees,
    )
    # sample datas
    samples, _ = sample_dataloader.__iter__().__next__()
    batches, batch_labels = batch_dataloader.__iter__().__next__()
    loss = net(samples,batches,batch_labels)

    return loss



def test(folder,net1,net2):
    total_rewards = 0
    for i in range(config.TEST_EPISODE):
        degrees = random.choice([0, 90, 180, 270])
        task = tg.OmniglotTask(
            folder,
            config.CLASS_NUM,
            config.SAMPLE_NUM_PER_CLASS,
            config.SAMPLE_NUM_PER_CLASS,
        )
        sample_dataloader = tg.get_data_loader(
            task,
            num_per_class=config.SAMPLE_NUM_PER_CLASS,
            split="train",
            shuffle=False,
            rotation=degrees,
        )
        test_dataloader = tg.get_data_loader(
            task,
            num_per_class=config.SAMPLE_NUM_PER_CLASS,
            split="test",
            shuffle=True,
            rotation=degrees,
        )

        sample_images, _ = sample_dataloader.__iter__().__next__()
        test_images, test_labels = test_dataloader.__iter__().__next__()

        # calculate features
        sample_features = net1(sample_images)  # 5x64
        test_features = net1(test_images)  # 20x64

        sample_features_ext = sample_features.unsqueeze(0).tile(
            (config.SAMPLE_NUM_PER_CLASS * config.CLASS_NUM, 1, 1, 1, 1)
        )
        test_features_ext = test_features.unsqueeze(0).tile(
            (config.SAMPLE_NUM_PER_CLASS * config.CLASS_NUM, 1, 1, 1, 1)
        )
        test_features_ext = ops.swapaxes(test_features_ext, 0, 1)

        relation_pairs = ops.cat(
            (sample_features_ext, test_features_ext), 2
        ).view(-1, config.FEATURE_DIM * 2, 5, 5)
        relations = net2(relation_pairs).view(-1, config.CLASS_NUM)
        _, predict_labels = ops.max(relations, 1)
        _, labels = ops.max(test_labels, 1)
        rewards = [
            1 if predict_labels[j] == labels[j] else 0
            for j in range(config.CLASS_NUM)
        ]
        total_rewards += np.sum(rewards)
    test_accuracy = total_rewards / 1.0 / config.CLASS_NUM / config.TEST_EPISODE
    return test_accuracy


@moxing_wrapper(pre_process=modelarts_pre_process)
def Train(train_foder,test_foder,train_net,net1,net2,net1_path,net2_path):
    print("Training...")
    last_accuracy = 0.0
    for episode in range(config.EPISODE):
        start_time = time.perf_counter()
        loss = train_one_step(net=train_net,folder=train_foder)
        train_time = time.perf_counter()
        if (episode + 1) % config.PRINT_FREQUENCY == 0:
            print("episode:{}    loss:{:.8f}    Time of train one episode:{:.3f}ms".format(episode + 1, loss.item(),(train_time-start_time) * 1000))
          
        if (episode + 1) % config.TEST_FREQUENCY == 0:
            print("Testing...") 
            test_time1 = time.perf_counter() 
            test_accuracy = test(folder=test_foder,net1=net1,net2=net2)
            test_time2 = time.perf_counter()
            print('Time of test one episode:{:.3f}ms'.format(((test_time2-test_time1)/config.TEST_EPISODE) * 1000))
            print("test accuracy:{:.4f}%\n".format(test_accuracy*100) )

            if test_accuracy > last_accuracy:
                # save networks
                ms.save_checkpoint(feature_encoder,net1_path)
                ms.save_checkpoint(relation_network,net2_path) 
                print("\nsave networks for episode:", episode)
                last_accuracy = test_accuracy


if __name__ == "__main__":
    Train(train_foder=metatrain_character_folders,
          test_foder=metatest_character_folders,
          train_net=train_net,
          net1=feature_encoder,
          net2=relation_network,
          net1_path=encoder_checkpoint,
          net2_path=relation_checkpoint
          )
