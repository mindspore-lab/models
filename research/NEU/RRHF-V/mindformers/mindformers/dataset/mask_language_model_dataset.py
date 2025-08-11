# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Masked Image Modeling Dataset."""
import os
import copy
from typing import Optional, Union, Callable

import mindspore.common.dtype as mstype
from mindspore.dataset.transforms import TypeCast
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map
from .dataloader import build_dataset_loader
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class MaskLanguageModelDataset(BaseDataset):
    """
    Bert pretrain dataset.

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        input_columns (list):
            Column name before the map function.
        batch_size (int):
            Size of each batch. Default: 8.
        drop_remainder (bool):
            Whether to discard the last batch when the number of data items contained
            in the last batch is smaller than batch_size. Default: True.
        num_parallel_workers (int):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: 8.
        repeat (int):
            Number of times this dataset is repeated. Default: 1.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            Indicates whether to use the NUMA binding function. Default: False.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        A dataset for MaskLanguageModelDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import MaskLanguageModelDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['fill_mask']['bert_tiny_uncased']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> # Note:
        >>> #     The detailed data setting could refer to
        >>> #     https://gitee.com/mindspore/mindformers/blob/dev/docs/task_cards/masked_image_modeling.md
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = MaskLanguageModelDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindspore.dataset import TFRecordDataset
        >>> from mindformers.dataset import MaskLanguageModelDataset
        >>> data_loader = TFRecordDataset(dataset_files="The required task dataset path",
        ...                               shuffle=True, shard_equal_rows=True)
        >>> dataset_from_param = MaskLanguageModelDataset(data_loader=data_loader, batch_size=1,
        ...                                               input_columns=['input_ids', 'input_mask', 'segment_ids',
        ...                                                              'next_sentence_labels', 'masked_lm_positions',
        ...                                                              'masked_lm_ids', 'masked_lm_weights'])
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                input_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        logger.info("Now Create Masked Image Modeling Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()
        dataset_config = copy.deepcopy(dataset_config)
        if isinstance(dataset_config.data_loader, dict):
            if not (dataset_config.data_loader.type == 'MindDataset' or
                    dataset_config.data_loader.type == 'TFRecordDataset'):
                raise NotImplementedError("Now, Causal Language Modeling Dataset only supports "
                                          "MindSpore's MindDataset and TFRecordDataset two data loading modes")
            dataset_files = MaskLanguageModelDataset._get_dataset_files(dataset_config)
            dataset = build_dataset_loader(dataset_config.data_loader,
                                           default_args={'dataset_files': dataset_files,
                                                         'num_shards': device_num,
                                                         'shard_id': rank_id})
        else:
            dataset = dataset_config.data_loader

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                output_columns=dataset_config.input_columns,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.project(columns=dataset_config.input_columns)
        dataset = dataset.repeat(dataset_config.repeat)
        type_cast_op = TypeCast(mstype.int32)
        for input_arg in dataset_config.input_columns:
            dataset = get_dataset_map(dataset, type_cast_op, input_columns=input_arg)
        return dataset

    @staticmethod
    def _get_dataset_files(dataset_config):
        """get dataset files."""
        dataset_files = []
        if dataset_config.data_loader.dataset_dir:
            data_dir = dataset_config.data_loader.pop("dataset_dir")
            if os.path.isdir(data_dir):
                for r, _, f in os.walk(data_dir):
                    for file in f:
                        if file.endswith(".mindrecord") or file.endswith(".tfrecord"):
                            dataset_files.append(os.path.join(r, file))
                dataset_files.sort()
            else:
                if data_dir.endswith(".mindrecord") or data_dir.endswith(".tfrecord"):
                    dataset_files = data_dir
        elif dataset_config.data_loader.dataset_files:
            dataset_files = dataset_config.data_loader.dataset_files
            if isinstance(dataset_files, (list, tuple)):
                dataset_files = list(dataset_files)
        else:
            raise ValueError(f"data_loader must contain dataset_dir or dataset_files,"
                             f"but get {dataset_config.data_loader}.")
        return dataset_files
