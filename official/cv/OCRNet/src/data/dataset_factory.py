import os
import cv2
import copy
import multiprocessing
from mindspore import dataset as ds
from .cityscapes import Cityscapes
from .transforms_factory import create_transform


def create_dataset(cfg, batch_size, num_parallel_workers=8, group_size=1, rank=0, task="train"):
    """
    Creates dataset by name.
    Args:
        cfg (obj): Configs
        batch_size (int): The number of rows each batch is created with.
        num_parallel_workers (int): Number of workers(threads) to process the dataset in parallel.
        group_size (int): Number of shards that the dataset will be divided
        rank (int): The shard ID within `group_size`
        is_train (bool): whether is training.
        task (str): 'train', 'eval' or 'infer'.
    """
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(2)
    ds.config.set_enable_shared_mem(True)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(num_parallel_workers, cores // group_size)
    ds.config.set_num_parallel_workers(num_parallel_workers)
    is_train = task == "train"
    if task == "train":
        trans_config = getattr(cfg, "train_transforms", cfg)
    elif task in ("val", "eval"):
        trans_config = getattr(cfg, "eval_transforms", cfg)
    else:
        raise NotImplementedError
    item_transforms = getattr(trans_config, "item_transforms", [])
    transforms_name_list = []
    for transform in item_transforms:
        transforms_name_list.extend(transform.keys())
    transforms_list = []
    for i, transform_name in enumerate(transforms_name_list):
        transform = create_transform(item_transforms[i])
        transforms_list.append(transform)
    dataset = None
    if cfg.name == "cityscapes":
        dataset = Cityscapes(ignore_label=cfg.ignore_label).create_dataset(
            dataset_dir=cfg.dataset_dir,
            map_label=cfg.map_label,
            group_size=group_size,
            rank=rank,
            is_train=is_train,
        )
    else:
        NotImplementedError
    if task == "train":
        dataset = dataset.map(
            operations=transforms_list, input_columns=["image", "label"], python_multiprocessing=True, max_rowsize=64
        )
        dataset = dataset.project(["image", "label"])
    else:
        dataset = dataset.map(
            operations=transforms_list, input_columns=["image", "label"], python_multiprocessing=True, max_rowsize=64
        )
        dataset = dataset.project(["image", "label"])

    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset, dataset.get_dataset_size()


def get_dataset(cfg):
    if cfg.name == "cityscapes":
        dataset = Cityscapes(ignore_label=cfg.ignore_label)
    else:
        NotImplementedError
    return dataset
