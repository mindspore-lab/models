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
from .cityscapes import cityscapesDataSet
from .cityscapes_self_distill import cityscapesSelfDistillDataSet
from .synthia import synthiaDataSet
from .gta5 import GTA5DataSet

class DatasetCatalog(object):
#     DATASET_DIR = "/data/zd/data/"
    DATASET_DIR = "./datasets/"

    DATASETS = {
        "gta5_train": {
            "data_dir": "gtav",
            "data_list": "gta5_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscape",
            "data_list": "cityscape/cityscapes_train_list.txt"
        },
        "cityscapes_self_distill_train": {
            "data_dir": "cityscape",
            "data_list": "cityscape/cityscapes_train_list.txt",
            "label_dir": "cityscapes/soft_labels_by_advd19/inference/cityscapes_train"
        },
        "cityscapes_val": {
            "data_dir": "cityscape",
            "data_list": "cityscape/cityscapes_val_list.txt"
        },
    }


    # DATASETS = {
    #     "gta5_train": {
    #         "data_dir": "GTA5/GTAV",
    #         "data_list": "GTA5/GTAV/gta5_train_list.txt"
    #     },
    #     "synthia_train": {
    #         "data_dir": "synthia",
    #         "data_list": "synthia_train_list.txt"
    #     },
    #     "cityscapes_train": {
    #         "data_dir": "cityscape/cityscapes/Cityscapes",
    #         "data_list": "cityscape/cityscapes/Cityscapes/cityscapes_train_list.txt"
    #     },
    #     "cityscapes_self_distill_train": {
    #         "data_dir": "cityscape/cityscapes/Cityscapes",
    #         "data_list": "cityscape/cityscapes/Cityscapes/cityscapes_train_list.txt",
    #         "label_dir": "cityscapes/soft_labels/inference/cityscapes_train"
    #     },
    #     "cityscapes_val": {
    #         "data_dir": "cityscape/cityscapes/Cityscapes",
    #         "data_list": "cityscape/cityscapes/Cityscapes/cityscapes_val_list.txt"
    #     },
    # }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None):
        if "gta5" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTA5DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                # data_dir = './datasets'
                # args['root'] = os.path.join(data_dir, attrs["data_dir"])
                # args['data_list'] = os.path.join(data_dir, attrs["data_list"])
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        raise RuntimeError("Dataset not available: {}".format(name))