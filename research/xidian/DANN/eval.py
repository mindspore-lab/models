# Copyright 2023 Xidian University
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

import os
import numpy as np
import mindspore.dataset as ds
import mindspore as ms
import mindspore.dataset.vision as c_vision
import mindspore.ops as ops
from mindspore import Tensor,context
from model_utils.config import config

from src.models import Class_classifier,  Backbone
from src.data_loader import GetLoader
from src.train_cell import withEvalCell

def test(dataset_name,path):
    assert dataset_name in ['MNIST', 'mnist_m']
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.join(current_dir,'models')
    print(model_root)
    image_root = os.path.join(current_dir,'dataset', dataset_name)
    print(image_root)
    dataset_dir = os.path.join(current_dir,"dataset/MNIST")
    batch_size = 64

    loss_class = ops.NLLLoss()

    """load data"""
    img_transform_source = [
        c_vision.RandomResize(28),
        c_vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        c_vision.HWC2CHW()]

    img_transform_target = [
        c_vision.RandomResize(28),
        c_vision.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        c_vision.HWC2CHW()]

    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
        dataset = GetLoader(
            data_root=os.path.join(image_root, 'mnist_m_test'),
            data_list=test_list,
        )
        dataset = ds.GeneratorDataset(source=dataset, column_names=["image", "label"])
        dataset = dataset.map(operations=img_transform_target, input_columns=["image"], output_columns=["image"])
    else:
        dataset = ds.MnistDataset(
            dataset_dir=dataset_dir,
            usage='test'
        )
        dataset = dataset.map(operations=img_transform_source)

    dataloader = dataset.batch(batch_size, drop_remainder=True)

    """test"""
    backbone1 = Backbone()
    class_classifier1 = Class_classifier()

    param_dict_backbone = ms.load_checkpoint(path)
    ms.load_param_into_net(backbone1, param_dict_backbone)
    param_dict_class_classifier = ms.load_checkpoint(os.path.join(model_root, 'best_class_classifier.ckpt'))
    ms.load_param_into_net(class_classifier1, param_dict_class_classifier)

    len_dataloader = dataloader.get_dataset_size()
    data_target_iter = dataloader.create_dict_iterator()

    n_total = 0
    n_correct = 0
    weightClass = Tensor(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float32))
    eval_net = withEvalCell(backbone1, class_classifier1, loss_class)
    eval_net.set_train(False)

    for i in range(len_dataloader):
        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target["image"], data_target["label"]
        t_img = Tensor(t_img, ms.float32)
        t_label = Tensor(t_label, ms.int32)

        batch_size = len(t_label)

        class_output, pred = eval_net(t_img, t_label, weightClass)
        pred = Tensor(pred, dtype=ms.int32)
        correct = ops.Equal()(pred, t_label)
        for i in range(len(correct)):
            if correct[i]:
                n_correct += 1
        n_total += batch_size

    accu = n_correct * 1.0 / n_total

    return accu

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backbone_ckpt_root = os.path.join(current_dir,"models/backbone_pretrain.ckpt")
    accu_s = test("MNIST", backbone_ckpt_root)
    print('source domain accuracy: %f' % accu_s)
    backbone_t_ckpt_root = os.path.join(current_dir, "models/best_backbone_t.ckpt")
    accu_t = test("mnist_m", backbone_t_ckpt_root)
    print('target domain accuracy: %f' % accu_t)