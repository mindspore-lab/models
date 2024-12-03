import logging
import multiprocessing
import os

import mindspore as ms

from .pgnet_dataset import PGDataset

__all__ = ["build_dataset"]
_logger = logging.getLogger(__name__)

supported_dataset_types = ["PGDataset"]

def build_dataset(
    dataset_config: dict,
    loader_config: dict,
    num_shards=None,
    shard_id=None,
    is_train=True,
    **kwargs,
):
    dataset_config = _check_dataset_paths(dataset_config)

    num_devices = 1 if num_shards is None else num_shards
    cores = multiprocessing.cpu_count()
    NUM_WORKERS_BATCH = 2
    NUM_WORKERS_MAP = int(
        cores / num_devices - NUM_WORKERS_BATCH
    )
    num_workers = loader_config.get("num_workers", NUM_WORKERS_MAP)
    if num_workers > int(cores / num_devices):
        _logger.warning(
            f"`num_workers` is adjusted to {int(cores / num_devices)} since {num_workers}x{num_devices} "
            f"exceeds the number of CPU cores {cores}"
        )
        num_workers = int(cores / num_devices)
    prefetch_size = loader_config.get("prefetch_size", 16) 
    ms.dataset.config.set_prefetch_size(prefetch_size)
    max_rowsize = loader_config.get("max_rowsize", 64)

    dataset_class_name = dataset_config.pop("type")
    assert dataset_class_name in supported_dataset_types, "Invalid dataset name"
    dataset_class = eval(dataset_class_name)
    dataset_args = dict(is_train=is_train, **dataset_config)
    if "use_minddata" in dataset_args and dataset_args["use_minddata"]:
        minddata_op_list = _parse_minddata_op(dataset_args)

    dataset = dataset_class(**dataset_args)

    dataset_column_names = dataset.get_output_columns()

    ds = ms.dataset.GeneratorDataset(
        dataset,
        column_names=dataset_column_names,
        num_parallel_workers=num_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        python_multiprocessing=True,
        max_rowsize=max_rowsize,
        shuffle=loader_config["shuffle"],
    )

    if "use_minddata" in dataset_args and dataset_args["use_minddata"]:
        ds = ds.map(
            operations=minddata_op_list,
            input_columns=["image"],
            num_parallel_workers=num_workers,
            python_multiprocessing=True,
        )

    # get batch of dataset by collecting batch_size consecutive data rows and apply batch operations
    num_samples = ds.get_dataset_size()
    batch_size = loader_config["batch_size"]

    device_id = 0 if shard_id is None else shard_id
    is_main_device = device_id == 0
    _logger.info(
        f"Creating dataloader (training={is_train}) for device {device_id}. Number of data samples: {num_samples}"
        f" per device ({num_samples * num_devices} total)."
    )

    if "refine_batch_size" in kwargs:
        if kwargs["refine_batch_size"]:
            batch_size = _check_batch_size(num_samples, batch_size, refine=kwargs["refine_batch_size"])

    drop_remainder = loader_config.get("drop_remainder", is_train)
    if is_train and drop_remainder is False and is_main_device:
        _logger.warning(
            "`drop_remainder` should be True for training, "
            "otherwise the last batch may lead to training fail in Graph mode"
        )

    if not is_train:
        if drop_remainder and is_main_device:
            _logger.warning(
                "`drop_remainder` is forced to be False for evaluation "
                "to include the last batch for accurate evaluation."
            )
            drop_remainder = loader_config.get("drop_remainder", False)

    dataloader = ds.batch(
        batch_size,
        drop_remainder=drop_remainder,
        num_parallel_workers=min(
            num_workers, 2
        ),
    )
    return dataloader

def _check_dataset_paths(dataset_config):
    if "dataset_root" in dataset_config:
        if isinstance(dataset_config["data_dir"], str):
            dataset_config["data_dir"] = os.path.join(
                dataset_config["dataset_root"], dataset_config["data_dir"]
            )
        else:
            dataset_config["data_dir"] = [
                os.path.join(dataset_config["dataset_root"], dd) for dd in dataset_config["data_dir"]
            ]
        if "label_file" in dataset_config:
            if dataset_config["label_file"]:
                if isinstance(dataset_config["label_file"], str):
                    dataset_config["label_file"] = os.path.join(
                        dataset_config["dataset_root"], dataset_config["label_file"]
                    )
                elif isinstance(dataset_config["label_file"], list):
                    dataset_config["label_file"] = [
                        os.path.join(dataset_config["dataset_root"], lf) for lf in dataset_config["label_file"]
                    ]

    return dataset_config

def _check_batch_size(num_samples, ori_batch_size=32, refine=True):
    if num_samples % ori_batch_size == 0:
        return ori_batch_size
    else:
        # search a batch size that is divisible by num samples.
        for bs in range(ori_batch_size - 1, 0, -1):
            if num_samples % bs == 0:
                _logger.info(
                    f"Batch size for evaluation is refined to {bs} to ensure the last batch will not be "
                    f"dropped/padded in graph mode."
                )
                return bs

def _parse_minddata_op(dataset_args):
    minddata_op_idx = []
    minddata_op_list = []
    for i, transform_dict in enumerate(dataset_args["transform_pipeline"]):
        if "NormalizeImage" in transform_dict.keys():
            from models.data.transforms.general_transforms import get_value

            normalize_transform = transform_dict["NormalizeImage"]
            mean = get_value(normalize_transform.get("mean", "imagenet"), "mean")
            std = get_value(normalize_transform.get("std", "imagenet"), "std")
            minddata_op_idx.append(i)
            normalize_op = ms.dataset.vision.Normalize(mean=mean, std=std)
            minddata_op_list.append(normalize_op)
            continue
        if "ToCHWImage" in transform_dict.keys():
            minddata_op_idx.append(i)
            change_swap_op = ms.dataset.vision.HWC2CHW()
            minddata_op_list.append(change_swap_op)
            continue
    for _ in range(len(minddata_op_idx)):
        dataset_args["transform_pipeline"].pop(minddata_op_idx.pop())
    return minddata_op_list
