from pathlib import Path
from models.data.builder import build_dataset

cur_dir = Path(__file__).resolve().parent
test_file_dir = cur_dir / "test_files"

DICT_PATHS_OCR = str(test_file_dir / "ic_dict.txt")
LABEL_PATHS_COR = str(test_file_dir / "train.txt")

def test_dataset():
    data_config = {
        "type": "PGDataset",
        "dataset_root": test_file_dir,
        "dataset": "totaltext",
        "data_dir": "test_data",
        "label_file": LABEL_PATHS_COR,
        "sample_ratio": 1.0,
        "transform_pipeline": [
            {
                "DecodeImage": {
                "img_mode": "BGR",
                "channel_first": False,
                }
            },
            {
                "E2ELabelEncodeTrain": {
                }
            },
            {
                "PGProcessTrain": {
                "batch_size": 1,
                "use_resize": True,
                "use_random_crop": False,
                "min_crop_size": 24,
                "min_text_size": 4,
                "max_text_size": 512,
                "point_gather_mode": "align",
                "max_text_length": 50,
                "max_text_nums": 30,
                "tcl_len": 64,
                "character_dict_path": DICT_PATHS_OCR,
                },
            },
        ],
        "output_columns": ['images',  'tcl_maps', 'tcl_label_maps', 'border_maps',
                           'direction_maps', 'training_masks', 'label_list', 'pos_list', 'pos_mask'],
        "net_input_column_index": [0],
        "label_column_index": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    loader_config = {
        "shuffle": False,
        "batch_size": 1,
        "drop_remainder": False,
        "num_workers": 1,
    }

    data_loader = build_dataset(data_config, loader_config, num_shards=1, shard_id=0, is_train=True)
    assert data_loader.get_dataset_size() == 1

    for data in data_loader.create_dict_iterator():
        assert data["images"].shape == (1, 3, 512, 512)
        assert data["tcl_maps"].shape == (1, 1, 128, 128)
        assert data["tcl_label_maps"].shape == (1, 1, 128, 128)
        assert data["border_maps"].shape == (1, 5, 128, 128)
        assert data["direction_maps"].shape == (1, 3, 128, 128)
        assert data["training_masks"].shape == (1, 1, 128, 128)
        assert data["label_list"].shape == (1, 30, 50, 1)
        assert data["pos_list"].shape == (1, 30, 64, 3)
        assert data["pos_mask"].shape == (1, 30, 64, 1)
