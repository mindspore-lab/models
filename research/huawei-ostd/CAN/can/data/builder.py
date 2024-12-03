import logging
import multiprocessing
import os

import mindspore as ms

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .det_dataset import DetDataset
from .rec_dataset import RecDataset
from .utils.collate_fn import *

__all__ = ["build_dataset"]
_logger = logging.getLogger(__name__)

supported_dataset_types = [
    "BaseDataset",
    "DetDataset",
    "RecDataset",
]

supported_collator_types = [
    "can_collator",
]


def build_dataset(
    dataset_config: dict,
    loader_config: dict,
    num_shards=None,
    shard_id=None,
    is_train=True,
    **kwargs,
):
    """
    Build dataset for training and evaluation.

    Args:
        dataset_config (dict): dataset parsing and processing configuartion containing the following keys
            - type (str): dataset class name, please choose from `supported_dataset_types`.
            - dataset_root (str): the root directory to store the (multiple) dataset(s)
            - data_dir (Union[str, List[str]]): directory to the data, which is a subfolder path related to
              `dataset_root`. For multiple datasets, it is a list of subfolder paths.
            - label_file (Union[str, List[str]], *optional*): file path to the annotation related to the `dataset_root`.
              For multiple datasets, it is a list of relative file paths. Not required if using LMDBDataset.
            - sample_ratio (float): the sampling ratio of dataset.
            - shuffle (boolean): whether to shuffle the order of data samples.
            - transform_pipeline (list[dict]): each element corresponds to a transform operation on image and/or label
            - output_columns (list[str]): list of output features for each sample.
            - net_input_column_index (list[int]): input indices for network forward func in output_columns
        loader_config (dict): dataloader configuration containing keys:
            - batch_size (int): batch size for data loader
            - drop_remainder (boolean): whether to drop the data in the last batch when the total of data can not be
              divided by the batch_size
            - num_workers (int): number of subprocesses used to fetch the dataset in parallel.
        num_shards (int, *optional*): num of devices for distributed mode
        shard_id (int, *optional*): device id
        is_train (boolean): whether it is in training stage
        **kwargs: optional args for extension. If `refine_batch_size=True` is given in kwargs, the batch size will be
            refined to be divisable to avoid
            droping remainding data samples in graph model, typically used for precise evaluation.

    Return:
        data_loader (Dataset): dataloader to generate data batch
    """
    # Check dataset paths (dataset_root, data_dir, and label_file) and update to absolute format
    dataset_config = _check_dataset_paths(dataset_config)

    # Set default multiprocessing params for data pipeline
    # num_parallel_workers: Number of subprocesses used to fetch the dataset, transform data, or load batch in parallel
    num_devices = 1 if num_shards is None else num_shards
    cores = multiprocessing.cpu_count()
    NUM_WORKERS_BATCH = 2
    NUM_WORKERS_MAP = int(
        cores / num_devices - NUM_WORKERS_BATCH
    )  # optimal num workers assuming all cpu cores are used in this job
    num_workers = loader_config.get("num_workers", NUM_WORKERS_MAP)
    if num_workers > int(cores / num_devices):
        _logger.warning(
            f"`num_workers` is adjusted to {int(cores / num_devices)} since {num_workers}x{num_devices} "
            f"exceeds the number of CPU cores {cores}"
        )
        num_workers = int(cores / num_devices)
    # prefetch_size: the length of the cache queue in the data pipeline for each worker, used to reduce waiting time.
    # Larger value leads to more memory consumption. Default: 16
    prefetch_size = loader_config.get("prefetch_size", 16)  #
    ms.dataset.config.set_prefetch_size(prefetch_size)
    # max_rowsize: MB of shared memory between processes to copy data. Only used when python_multiprocessing is True.
    max_rowsize = loader_config.get("max_rowsize", 64)
    # auto tune num_workers, prefetch. (This conflicts the profiler)
    # ms.dataset.config.set_autotune_interval(5)
    # ms.dataset.config.set_enable_autotune(True, "./dataproc_autotune_out")

    # 1. create source dataset (GeneratorDataset)
    # Invoke dataset class
    dataset_class_name = dataset_config.pop("type")
    assert dataset_class_name in supported_dataset_types, "Invalid dataset name"
    dataset_class = eval(dataset_class_name)
    dataset_args = dict(is_train=is_train, **dataset_config)
    if "use_minddata" in dataset_args and dataset_args["use_minddata"]:
        minddata_op_list = _parse_minddata_op(dataset_args)

    dataset = dataset_class(**dataset_args)

    dataset_column_names = dataset.get_output_columns()

    # Generate source dataset (source w.r.t. the dataset.map pipeline)
    # based on python callable numpy dataset in parallel
    ds = ms.dataset.GeneratorDataset(
        dataset,
        column_names=dataset_column_names,
        num_parallel_workers=num_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        python_multiprocessing=True,  # keep True to improve performance for heavy computation.
        max_rowsize=max_rowsize,
        shuffle=loader_config["shuffle"],
    )

    # 2. data mapping using minddata C lib (optional)
    if "use_minddata" in dataset_args and dataset_args["use_minddata"]:
        ds = ds.map(
            operations=minddata_op_list,
            input_columns=["image"],
            num_parallel_workers=num_workers,
            python_multiprocessing=True,
        )

    # 3. create loader
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

    collate_fn = None
    if "collate_fn" in loader_config and loader_config["collate_fn"]:
        assert loader_config["collate_fn"] in supported_collator_types, "Invalid collator name"
        collate_fn = eval(loader_config["collate_fn"])
        dataloader = ds.batch(
            batch_size,
            drop_remainder=drop_remainder,
            num_parallel_workers=min(
                num_workers, 2
            ),  # set small workers for lite computation. TODO: increase for batch-wise mapping
            # input_columns=["image","label"],
            output_columns=loader_config["output_columns"],
            per_batch_map=collate_fn, # uncomment to use inner-batch transformation
        )

    if collate_fn is None:
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
            )  # to absolute path
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
        if "RandomColorAdjust" in transform_dict.keys():
            minddata_op_idx.append(i)
            color_adjust_op = ms.dataset.vision.RandomColorAdjust(
                brightness=transform_dict["RandomColorAdjust"]["brightness"],
                saturation=transform_dict["RandomColorAdjust"]["saturation"],
            )
            minddata_op_list.append(color_adjust_op)
            continue
        if "NormalizeImage" in transform_dict.keys():
            from can.data.transforms.general_transforms import get_value

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
