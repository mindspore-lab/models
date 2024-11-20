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
import os
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import ops as ops
from mindspore import Tensor as Tensor
from mindspore.dataset import vision, transforms


def mnist_dataloader(batch_size, train):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = current_dir + "/dataset/MNIST_Data/"
    if train:
        root = root + 'train'
    else:
        root = root + 'test'

    dataset = ds.MnistDataset(dataset_dir=root)
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    dataloader = ds.GeneratorDataset(
        dataset, column_names=["image", 'label'], shuffle=train)

    return dataloader


def svhn_dataloader(batch_size, split='train'):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    svhn_dataset_dir = current_dir + "/dataset/SVHN"
    dataset = ds.SVHNDataset(dataset_dir=svhn_dataset_dir, usage=str(split))

    image_transforms = [
        vision.Resize((28, 28)),
        vision.ToPIL(),
        vision.Grayscale(),
        vision.ToTensor(),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    dataloader = ds.GeneratorDataset(
        dataset, column_names=["image", 'label'], shuffle=True)
    return dataloader


def sample_data():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = current_dir + "/dataset/MNIST_Data/train"
    dataset = ds.MnistDataset(dataset_dir=root)
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')

    n = dataset.get_dataset_size()

    X = ms.Tensor(np.zeros((n, 1, 28, 28)), ms.float32)
    Y = ms.Tensor(np.zeros((n)), ms.float32)

    dataloader = ds.GeneratorDataset(dataset.batch(600), column_names=[
                                     "image", 'label'], shuffle=True)
    for i, item in enumerate(dataloader):
        X[i*600:(i+1)*600] = item[0]
        Y[i*600:(i+1)*600] = item[1]
    return X, Y.astype(ms.int32)


def create_target_samples(n=1):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    svhn_dataset_dir = current_dir + "/dataset/SVHN"
    dataset = ds.SVHNDataset(dataset_dir=svhn_dataset_dir, usage='train')
    image_transforms = [
        vision.Resize((28, 28)),
        vision.ToPIL(),
        vision.Grayscale(),
        vision.ToTensor(),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
    ]
    label_transform = transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    X, Y = [], []
    classes = 10*[n]
    for i, data in enumerate(dataset):
        if len(X) == n*10:
            break
        x, y = data[0], data[1]
        if classes[y] > 0:
            X.append(x)
            Y.append(y.asnumpy())
            classes[y] -= 1
    assert (len(X) == n*10)

    return ops.stack(X, axis=0), Tensor(np.array(Y))


"""
G1: a pair of pic comes from same domain ,same class
G3: a pair of pic comes from same domain, different classes

G2: a pair of pic comes from different domain,same class
G4: a pair of pic comes from different domain, different classes
"""


def create_groups(X_s, Y_s, X_t, Y_t, seed=1):
    # change seed so every time wo get group data will different in source domain,but in target domain, data not change
    ms.set_seed(1 + seed)
    n = X_t.shape[0]  # 10*shot

    # shuffle order
    classes, _ = ops.Unique()(Y_t)
    inds = np.random.permutation(classes.shape[0])
    classes = classes.asnumpy()[inds]
    class_num = classes.shape[0]
    shot = n//class_num

    def s_idxs(c):
        equal = ops.Equal()
        idx = ops.nonzero(equal(Y_s, int(c))).asnumpy()
        return idx[np.random.permutation(idx.shape[0])][:shot*2].squeeze()

    def t_idxs(c):
        equal = ops.Equal()
        idx = ops.nonzero(equal(Y_t, int(c))).asnumpy()
        return idx[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    target_idxs = list(map(t_idxs, classes))

    source_matrix = Tensor(source_idxs)
    target_matrix = Tensor(target_idxs)

    G1, G2, G3, G4 = [], [], [], []
    Y1, Y2, Y3, Y4 = [], [], [], []

    for i in range(10):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j*2]],
                      X_s[source_matrix[i][j*2+1]]))
            Y1.append((Y_s[source_matrix[i][j*2]],
                      Y_s[source_matrix[i][j*2+1]]))
            G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
            G3.append((X_s[source_matrix[i % 10][j]],
                      X_s[source_matrix[(i+1) % 10][j]]))
            Y3.append((Y_s[source_matrix[i % 10][j]],
                      Y_s[source_matrix[(i + 1) % 10][j]]))
            G4.append((X_s[source_matrix[i % 10][j]],
                      X_t[target_matrix[(i+1) % 10][j]]))
            Y4.append((Y_s[source_matrix[i % 10][j]],
                      Y_t[target_matrix[(i + 1) % 10][j]]))

    groups = [G1, G2, G3, G4]
    groups_y = [Y1, Y2, Y3, Y4]

    # make sure we sampled enough samples
    for g in groups:
        assert (len(g) == n)
    return groups, groups_y


def sample_groups(X_s, Y_s, X_t, Y_t, seed=1):
    return create_groups(X_s, Y_s, X_t, Y_t, seed=seed)
