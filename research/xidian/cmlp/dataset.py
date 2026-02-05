import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset.transforms.transforms import TypeCast
from model.mlp import ConvMLP
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms import c_transforms
import mindspore.dataset.transforms as transforms
def create_cifar_dataset( do_train,dataset_choose='cifar10', batch_size=128, image_size=(224, 224), mixup=False,numwork=8):
    if dataset_choose=='cifar10':
        dataset_path = './data/cifar-10-batches-bin'
        if do_train:
            data_set = ds.Cifar10Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork,
                                         num_shards=1, shard_id=0,usage='train')
        else:
            data_set = ds.Cifar10Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork,
                                         num_shards=1, shard_id=0,usage='test')
    elif dataset_choose=='cifar100':
        dataset_path = './data/cifar-100-binary'
        if do_train:

            data_set = ds.Cifar100Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork,
                                          num_shards=1, shard_id=0,usage='train')
        else:

            data_set = ds.Cifar100Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork,
                                          num_shards=1, shard_id=0,usage='test')

    # define map operations
    if mixup==False:
        trans = []
        trans += [
            vision.Resize(image_size),
            vision.Rescale(1.0 / 255.0, 0.0),

        ]
        if do_train:
            trans += [
                # vision.RandomCrop((32, 32), (4, 4, 4, 4)),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.RandomVerticalFlip(prob=0.5)
            ]

        trans += [

            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            vision.HWC2CHW(),
        ]

        type_cast_op = TypeCast(ms.int32)

        data_set = data_set.map(operations=type_cast_op, input_columns="label")
        data_set = data_set.map(operations=trans, input_columns="image")
        data_set = data_set.batch(batch_size, drop_remainder=do_train)
    elif mixup==True:
        trans = []
        trans += [
            vision.Resize(image_size),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            vision.HWC2CHW(),
        ]

        type_cast_op = TypeCast(ms.float32)

        # data_set = dataset.map(operations=type_cast_op, input_columns="label")

        onehot_op = transforms.OneHot(num_classes=10)
        data_set = data_set.map(operations=trans, input_columns="image")
        data_set = data_set.batch(batch_size, drop_remainder=do_train)
        data_set  = data_set.map(operations=onehot_op,input_columns=["label"])
        data_set=data_set.map(operations=vision.MixUp(batch_size=batch_size, alpha=0.8),
        input_columns=["image", "label"])
        data_set = data_set.map(operations=type_cast_op, input_columns="image")
        data_set = data_set.map(operations=type_cast_op, input_columns="label")
    # apply batch operations

    return data_set

def create_cifar_datasetmul( do_train,dataset_choose='cifar10', batch_size=128, image_size=(224, 224), mixup=False,numwork=8, rank_id=0, rank_size=1):
    if dataset_choose=='cifar10':
        if do_train:
            dataset_path='./data/cifar-10-batches-bin'
            data_set = ds.Cifar10Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork, num_shards=rank_size, shard_id=rank_id)
        else:
            dataset_path = './data/cifar-10-val-bin'
            data_set = ds.Cifar10Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork, num_shards=rank_size, shard_id=rank_id)
    elif dataset_choose=='cifar100':
        if do_train:
            dataset_path='./data/cifar-100-binary'
            data_set = ds.Cifar100Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork, num_shards=rank_size, shard_id=rank_id)
        else:
            dataset_path = './data/cifar-100-val-binary'
            data_set = ds.Cifar100Dataset(dataset_path, shuffle=do_train,num_parallel_workers=numwork, num_shards=rank_size, shard_id=rank_id)

    # define map operations
    if mixup==False:
        trans = []
        trans += [
            vision.Resize(image_size),
            vision.Rescale(1.0 / 255.0, 0.0),

        ]
        if do_train:
            trans += [
                # vision.RandomCrop((32, 32), (4, 4, 4, 4)),
                vision.RandomHorizontalFlip(prob=0.5),
                vision.RandomVerticalFlip(prob=0.5)
            ]

        trans += [

            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            vision.HWC2CHW(),
        ]

        type_cast_op = TypeCast(ms.int32)

        data_set = data_set.map(operations=type_cast_op, input_columns="label")
        data_set = data_set.map(operations=trans, input_columns="image")
        data_set = data_set.batch(batch_size, drop_remainder=do_train)
    elif mixup==True:
        trans = []
        trans += [
            vision.Resize(image_size),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
            vision.HWC2CHW(),
        ]

        type_cast_op = TypeCast(ms.float32)

        # data_set = dataset.map(operations=type_cast_op, input_columns="label")

        onehot_op = transforms.OneHot(num_classes=10)
        data_set = data_set.map(operations=trans, input_columns="image")
        data_set = data_set.batch(batch_size, drop_remainder=do_train)
        data_set  = data_set.map(operations=onehot_op,input_columns=["label"])
        data_set=data_set.map(operations=vision.MixUp(batch_size=batch_size, alpha=0.8),
        input_columns=["image", "label"])
        data_set = data_set.map(operations=type_cast_op, input_columns="image")
        data_set = data_set.map(operations=type_cast_op, input_columns="label")
    # apply batch operations

    return data_set
