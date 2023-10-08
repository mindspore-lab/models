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
from scipy.io import loadmat
import numpy as np
import argparse
from mindspore.mindrecord import FileWriter


def parse_args():
    """Get arguments from command-line."""
    parser = argparse.ArgumentParser(description='MindSpore MCD MindRecord make.')

    parser.add_argument("--svhn_train_path", type=str, default="./data/svhn/train_32x32.mat",
                        help="Storage path of svhn.")
    parser.add_argument("--svhn_test_path", type=str, default="./data/svhn/test_32x32.mat",
                        help="Storage path of svhn.")
    parser.add_argument("--mnist_data_path", type=str, default="./data/mnist_data.mat", help="Storage path of mnist.")

    return parser.parse_args()


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot


def load_svhn(train_path='./data/svhn/train_32x32.mat', test_path='./data/svhn/test_32x32.mat'):
    svhn_train = loadmat(train_path)
    svhn_test = loadmat(test_path)

    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test


def load_mnist(mnist_path='./data/mnist_data.mat', scale=True, usps=False, all_use=False):
    mnist_data = loadmat(mnist_path)
    if scale:
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test = mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))
    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    if usps and all_use != 'yes':
        mnist_train = mnist_train[:2000]
        train_label = train_label[:2000]

    return mnist_train, train_label, mnist_test, test_label


def read_data(svhn_train_path='./data/svhn/train_32x32.mat', svhn_test_path='./data/svhn/test_32x32.mat',
              mnist_data_path='./data/mnist_data.mat'):
    S = {}
    S_test = {}
    T = {}
    T_test = {}

    train_source, s_label_train, test_source, s_label_test = load_svhn(svhn_train_path, svhn_test_path)
    train_target, t_label_train, test_target, t_label_test = load_mnist(mnist_data_path)
    S['imgs'] = np.uint8(np.asarray(train_source.transpose(0, 2, 3, 1)))
    S['labels'] = np.int32(s_label_train)
    T['imgs'] = np.uint8(np.asarray(train_target.transpose(0, 2, 3, 1)))
    T['labels'] = np.int32(t_label_train)

    # input target samples for both
    S_test['imgs'] = np.uint8(np.asarray(test_source.transpose(0, 2, 3, 1)))
    S_test['labels'] = np.int32(s_label_test)
    T_test['imgs'] = np.uint8(np.asarray(test_target.transpose(0, 2, 3, 1)))
    T_test['labels'] = np.int32(t_label_test)

    source_set = data2list(S['imgs'], S['labels'], 'S')
    target_set = data2list(T['imgs'], T['labels'], 'T')

    source_test_set = data2list(S_test['imgs'], S_test['labels'], 'S')
    target_test_set = data2list(T_test['imgs'], T_test['labels'], 'T')

    return source_set, target_set, source_test_set, target_test_set


def data2list(imgs, label, use='S'):
    if use == 'S':
        name1 = 'S'
        name2 = 'S_label'
    elif use == 'T':
        name1 = 'T'
        name2 = 'T_label'

    num = imgs.shape[0]
    res_list = []
    for i in range(num):
        temp_row = {}
        temp_row[name1] = imgs[i].tobytes()
        temp_row[name2] = label[i]
        res_list.append(temp_row)

    return res_list


def mind_save(data, schema_json, name):
    # 'S' 'S_label' 'T' 'T_label'
    writer = FileWriter(file_name='./MindRecord/' + f'{name}' + '.mindrecord', shard_num=1, overwrite=True)
    writer.add_schema(schema_json, "index_schema")
    writer.write_raw_data(data)
    writer.commit()
    return


if __name__ == '__main__':
    args = parse_args()
    source_set, target_set, source_test_set, target_test_set = read_data(args.svhn_train_path, args.svhn_test_path,
                                                                         args.mnist_data_path)

    source_schema_json = {
        "S": {"type": "bytes"},
        "S_label": {"type": "int32"}
    }

    target_schema_json = {
        "T": {"type": "bytes"},
        "T_label": {"type": "int32"}
    }

    mind_save(source_set, source_schema_json, 'source_train')
    mind_save(target_set, target_schema_json, 'target_train')
    mind_save(source_test_set, source_schema_json, 'source_test')
    mind_save(target_test_set, target_schema_json, 'target_test')
