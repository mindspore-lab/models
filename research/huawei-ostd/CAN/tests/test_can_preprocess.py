import sys
from pathlib import Path

import mindspore as ms
from can.data.transforms.transforms_factory import create_transforms, run_transforms
from can.data import build_dataset

sys.path.append(".")
ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_files"

#This specifies the address of the data set used by the test case, which runs the test directly
TEST_FILE_OCR = str(test_file_dir)
DICT_PATHS_OCR = str(test_file_dir / "dict.txt")
LABEL_PATHS_OCR = str(test_file_dir / "label.txt")
IMG_PATHS_OCR = str(test_file_dir / "test_img/test_img.jpg")
IMG_FILE_PATHS = str(test_file_dir / "test_img")


def test_transform_pipeline():
    """
        Test description:
        This test case is used to test whether the model output preprocessing pipeline executes correctly.
        Used to test create_transforms and run_transforms interfaces.

        Modify variables:
        In this test case, you need to modify the following configurations based on the local file path
        1. character_dict_path
        2. img_path
        3. label
    """

    transform_pipeline = [
            {"DecodeImage": {
                "img_mode": "BGR",
                "channel_first": False,
                },
            },
            {"CANImageNormalize": {
                "mean": [0,0,0],
                "std": [1,1,1],
                "order": 'hwc',
                },
            },
            {"GrayImageChannelFormat": {
                "inverse": True
                },
            },
            {"CANLabelEncode":{
                "task": "train",
                "lower": False,
                "character_dict_path": DICT_PATHS_OCR
                }
            },
        ]

    data = {
        "img_path": IMG_PATHS_OCR,
        "label": "S = ( \sum _ { i = 1 } ^ { n } \theta _ { i } - ( n - 2 ) \pi ) r ^ { 2 }"
    }

    global_config = dict(is_train=True, use_minddata=False)
    transforms = create_transforms(transform_pipeline, global_config)
    data = run_transforms(data, transforms=transforms)


def test_train_data_config():
    """
        Test description:
        This test case is used to test whether the training data set can be loaded correctly.
        Because the test output will change as the data set changes, specific amounts can be 
        printed out for comparison if needed

        Modify variables:
        In this test case, you need to modify the following configurations based on the local file path
        1. character_dict_path
        2. dataset_root
        3. data_dir
        4. label_file
    """

    data_config = {
        "type": "RecDataset",
        "dataset_root": TEST_FILE_OCR,
        "data_dir": IMG_FILE_PATHS,
        "label_file": LABEL_PATHS_OCR,
        "sample_ratio": 1.0,
        "shuffle": False,
        "transform_pipeline": [
            {"DecodeImage": {
                "img_mode": "BGR",
                "channel_first": False,
                },
            },

            {"CANImageNormalize": {
                "mean": [0,0,0],
                "std": [1,1,1],
                "order": 'hwc',
                },
            },

            {"GrayImageChannelFormat": {
                "inverse": True
                },
            },

            {"CANLabelEncode":{
                "task": "train",
                "lower": False,
                "character_dict_path": DICT_PATHS_OCR
                }
            },
        ],

        "output_columns": ["image", "label"],
        "net_input_column_index": [0,1,2],
        "label_column_index": [2,3],
    }

    loader_config = {
        "shuffle": False,
        "batch_size": 1,
        "drop_remainder": False,
        "num_workers": 1,
        "collate_fn": "can_collator",
        "output_columns": ["images", "image_masks", "labels", "label_masks"],
    }
    data_loader = build_dataset(data_config, loader_config, num_shards=1, shard_id=0, is_train=True)

    print(f"size={data_loader.get_dataset_size()}")
    print(f"col_name={data_loader.get_col_names()}")   

    for data in data_loader.create_dict_iterator():
        print("----------")
        print(type(data))
        print(type(data["images"]))
        print(data["images"].shape)
        print(data["image_masks"].shape)
        print(data["labels"].shape)
        print(data["label_masks"].shape)


def test_eval_data_config():
    """
        Test description:
        This test case is used to test whether the eval data set can be loaded correctly.
        Because the test output will change as the data set changes, specific amounts can be 
        printed out for comparison if needed

        Modify variables:
        In this test case, you need to modify the following configurations based on the local file path
        1. character_dict_path
        2. dataset_root
        3. data_dir
        4. label_file
    """

    data_eval_config = {
        "type": "RecDataset",
        "dataset_root": TEST_FILE_OCR,
        "data_dir": IMG_FILE_PATHS,
        "label_file": LABEL_PATHS_OCR,
        "sample_ratio": 1.0,
        "shuffle": False,
        "transform_pipeline": [
            {"DecodeImage": {
                "img_mode": "BGR",
                "channel_first": False,
                },
            },

            {"CANImageNormalize": {
                "mean": [0,0,0],
                "std": [1,1,1],
                "order": 'hwc',
                },
            },

            {"GrayImageChannelFormat": {
                "inverse": True
                },
            },

            {"CANLabelEncode":{
                "task": "eval",
                "lower": False,
                "character_dict_path": DICT_PATHS_OCR
                }
            },
        ],

        "output_columns": ["image", "image_mask","ones_label","label","label_len"],
        "net_input_column_index": [0, 1, 2],
        "label_column_index": [3, 4],
    }

    eval_loader_config = {
        "shuffle": False,
        "batch_size": 1,
        "drop_remainder": False,
        "num_workers": 1,
    }

    data_eval_loader = build_dataset(data_eval_config, eval_loader_config, num_shards=1, shard_id=0, is_train=False)

    print(f"size={data_eval_loader.get_dataset_size()}")
    print(f"col_name={data_eval_loader.get_col_names()}")

    for data in data_eval_loader.create_dict_iterator():
        print(data["image"].shape)
        print(data["image_mask"].shape)
        print(data["ones_label"].shape)
        print(data["label"])
        print(data["label_len"])


if __name__ == "__main__":
    """
        Select the test function as needed
    """

    test_transform_pipeline()
    # test_train_data_config()
    # test_eval_data_config()
