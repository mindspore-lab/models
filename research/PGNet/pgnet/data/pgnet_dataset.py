import logging
import os
import random
from pathlib import Path

from .base_dataset import BaseDataset
from .transforms.transforms_factory import create_transforms, run_transforms

__all__ = ["PGDataset"]
_logger = logging.getLogger(__name__)

class PGDataset(BaseDataset):
    img_id_map = {
        "totaltext": 3,
        "icdar2015": 3,
    }

    def __init__(self, is_train=True, data_dir=None, label_file=None, dataset=None, sample_ratio=1.0, need_reset=True, transform_pipeline=None, output_columns=None, **kwargs):
        super().__init__(data_dir=data_dir, label_file=label_file, output_columns=output_columns)
        self.is_train = is_train

        self.sample_ratio = [sample_ratio] * len(self.label_file) if isinstance(sample_ratio, float) else sample_ratio

        if transform_pipeline is not None:
            global_config = dict(is_train=is_train)
            self.transforms = create_transforms(transform_pipeline, global_config)
        else:
            raise ValueError("No transform pipeline is specified!")

        if need_reset:
            self.need_reset = any(x < 1 for x in self.sample_ratio)
        else:
            self.need_reset = False

        self.dataset = [dataset] if isinstance(dataset, str) else dataset
        self.reset(output_columns)

    def reset(self, output_columns):
        self.data_list = self.load_data_list(self.label_file)
        print(f"reset dataset: {self.__class__.__name__}")

        for _data in self.data_list:
            try:
                _data = _data.copy()
                _data = run_transforms(_data, transforms=self.transforms)
                _available_keys = list(_data.keys())
                if _available_keys:
                    break
            except Exception:
                pass
        else:
            raise ValueError("All data cannot pass transforms")

        if output_columns is None:
            if self.output_columns is None:
                self.output_columns = _available_keys
        else:
            self.output_columns = []
            for k in output_columns:
                if k in _data:
                    self.output_columns.append(k)
                else:
                    raise ValueError(
                        f"Key '{k}' does not exist in data (available keys: {_data.keys()}). "
                        "Please check the name or the completeness transformation pipeline."
                    )

    def load_data_list(self, label_file, **kwargs):
        data_list = []
        for idx, label_fp in enumerate(label_file):
            img_dir = self.data_dir[idx]
            dataset = self.dataset[idx]
            with open(label_fp, "r", encoding="utf-8") as f:
                lines = f.readlines()
                sample_ratio = self.sample_ratio[idx]
                if sample_ratio == 1:
                    lines_idx = range(len(lines))
                elif self.need_reset:
                    lines_idx = random.sample(range(len(lines)), k=round(len(lines) * sample_ratio))
                else:
                    lines_idx = range(int(len(lines) * sample_ratio))
                for idx in lines_idx:
                    line = lines[idx]
                    img_name, annot_str = self._parse_annotation(line)

                    img_path = os.path.join(img_dir, img_name)
                    assert os.path.exists(img_path), "{} does not exist!".format(img_path)

                    stem = Path(img_name).stem
                    try:
                        img_id = int(stem[self.img_id_map.get(dataset, 0) :])
                    except Exception:
                        img_id = 0

                    data = {"img_path": img_path, "label": annot_str, "img_id": img_id}
                    data_list.append(data)
        return data_list

    def _parse_annotation(self, data_line):
        data_line_tmp = data_line.strip()
        if "\t" in data_line_tmp:
            img_name, annot_str = data_line_tmp.split("\t")
        elif " " in data_line_tmp:
            img_name, annot_str = data_line_tmp.split(" ")
        else:
            raise ValueError(
                "Incorrect label file format: the file name and the label should be separated by a space or tab"
            )
        return img_name, annot_str

    def __getitem__(self, index):
        data = self.data_list[index].copy()

        try:
            data = run_transforms(data, transforms=self.transforms)
            output_tuple = tuple(data[k] for k in self.output_columns)
        except Exception as e:
            if not self.is_train:
                _logger.warning(f"Error occurred while processing the image: {self.data_list[index]['img_path']}\n {e}")
                raise ValueError()
            return self[random.randrange(len(self.data_list))]

        return output_tuple
