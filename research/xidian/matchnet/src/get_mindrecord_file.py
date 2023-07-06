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
"""Make mindrecord file."""
import os
from PIL import Image
import numpy as np

from mindspore.mindrecord import FileWriter


def mind_save(dataset_name='liberty', data_dir='./Dataset/', is_train=True):
    """
    dataset: dataset:(image_file, matches_files)
    """
    if is_train:
        usage = 'train'
    else:
        usage = 'test'

    schema_json = {"image1": {"type": "bytes"}, "image2": {"type": "bytes"}, "matches": {"type": "int32"}}

    data = generate_dict_dataset(dataset_name, data_dir, is_train)

    writer = FileWriter(file_name='./MindRecord/' + f'{dataset_name}-{usage}' + '.mindrecord', shard_num=1,
                        overwrite=True)
    writer.add_schema(schema_json, "index_schema")
    # writer.add_index(indexes)
    writer.write_raw_data(data)
    writer.commit()
    return


def read_image_file(data_dir: str, image_ext: str, n: int):
    """Return a Tensor containing the patches"""

    def PIL2array(_img: Image.Image) -> np.ndarray:
        """Convert PIL image type to numpy 2D array"""
        return np.array(_img.getdata(), dtype=np.uint8).reshape(64, 64)

    def find_files(_data_dir: str, _image_ext: str):
        """Return a list with the file names of the images containing the patches"""
        files = []
        # find those files with the specified extension
        for file_dir in os.listdir(_data_dir):
            if file_dir.endswith(_image_ext):
                files.append(os.path.join(_data_dir, file_dir))
        return sorted(files)  # sort files in ascend order to keep relations

    patches = []
    list_files = find_files(data_dir, image_ext)

    for fpath in list_files:
        img = Image.open(fpath)
        for y in range(0, img.height, 64):
            for x in range(0, img.width, 64):
                patch = img.crop((x, y, x + 64, y + 64))
                # change to (C,H,W)
                patches.append(PIL2array(patch))
    return np.array(patches[:n])


def read_matches_files(data_dir: str, matches_file: str):
    """Return a Tensor containing the ground truth matches
    Read the file and keep only 3D point ID.
    Matches are represented with a 1, non matches with a 0.
    """
    matches = []
    with open(os.path.join(data_dir, matches_file)) as f:
        for line in f:
            line_split = line.split()
            matches.append([int(line_split[0]), int(line_split[3]), int(line_split[1] == line_split[4])])
    return matches


def generate_dict_dataset(dataset_name: str, data_dir, is_train: bool):
    """generate a list with each row of a dict contains image1, image2 and matches"""
    lens = {
        "notredame": 468159,
        "yosemite": 633587,
        "liberty": 450092
    }
    image_ext = "bmp"

    if is_train:
        matches_file = "m50_500000_500000_0.txt"
        matches_num = 500000
    else:
        matches_file = "m50_100000_100000_0.txt"
        matches_num = 100000
    data_set = data_dir + '/' + dataset_name
    patches = read_image_file(data_set, image_ext, lens[dataset_name])
    matches_info = read_matches_files(data_set, matches_file)
    res_list = []
    for i in range(matches_num):
        temp_row = {}
        m = matches_info[i]
        temp_row['image1'] = patches[m[0]].tobytes()
        temp_row['image2'] = patches[m[1]].tobytes()
        temp_row['matches'] = m[2]
        res_list.append(temp_row)
    return res_list


if __name__ == '__main__':
    mind_save(dataset_name='liberty', data_dir='./Dataset/', is_train=True)
    mind_save(dataset_name='liberty', data_dir='./Dataset/', is_train=False)

    mind_save(dataset_name='notredame', data_dir='./Dataset/', is_train=True)
    mind_save(dataset_name='notredame', data_dir='./Dataset/', is_train=False)

    mind_save(dataset_name='yosemite', data_dir='./Dataset/', is_train=True)
    mind_save(dataset_name='yosemite', data_dir='./Dataset/', is_train=False)