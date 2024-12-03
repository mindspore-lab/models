# Copyright 2020 Huawei Technologies Co., Ltd
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
'''python dataset.py'''

import os
import re

import pandas
import pandas as pd
import numpy as np
import librosa
from mindspore.mindrecord import FileWriter, FileReader
from mindspore import context
from src.model_utils.config import config as cfg
from src.model_utils.device_adapter import get_device_id
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def compute_melgram(audio_path, save_path='', filename='', save_npy=True):
    """
    extract melgram feature from the audio and save as numpy array

    Args:
        audio_path (str): path to the audio clip.
        save_path (str): path to save the numpy array.
        filename (str): filename of the audio clip.

    Returns:
        numpy array.

    """
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    try:
        src, _ = librosa.load(os.path.abspath(audio_path), sr=SR)  # whole signal
    except EOFError:
        print('File was damaged: ', audio_path)
        print('Now skip it!')
        return
    except FileNotFoundError:
        print('Failed to load the file: ', audio_path)
        print('Now skip it!')
        return
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample - n_sample_fit) // 2:(n_sample + n_sample_fit) //
                                                 2]
    logam = librosa.core.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(
        melgram(y=src, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS))
    ret = ret[np.newaxis, np.newaxis, :]
    # raise None
    if save_npy:
        save_path = save_path + filename[:-4] + '.npy'
        np.save(save_path, ret)


def get_data(features_data, labels_data):
    data_list = []
    for i, (label, feature) in enumerate(zip(labels_data, features_data)):
        data_json = {"id": i, "feature": feature, "label": label}
        data_list.append(data_json)
    return data_list


def convert(s):
    if s.isdigit():
        return int(s)
    return s


def extract_keywords_from_filename(filename):
    """
    Extracts keywords from a filename with a pattern like [keyword1][keyword2]...[keywordN]numbers__numbers.wav.

    Args:
    - filename (str): The filename from which to extract keywords.

    Returns:
    - list: A list of extracted keywords.
    """
    # 正则表达式匹配文件名中的关键词
    pattern = r"\[([^\]]+)\]"
    matches = re.findall(pattern, filename)
    return matches


def GetLabel_UrbanSound8K(info_path, cfg):
    """
    separate dataset into training set and validation set

    Args:
        audio_path (str): path to the information file.

    """

    label_list = {'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'}
    label_map = {key: label for label, key in enumerate(label_list)}
    info_file = os.path.join(info_path, 'UrbanSound8K.csv')

    def get_data(data_df: pandas.DataFrame, prefix=None):
        assert 'slice_file_name' in data_df.columns and 'fold' in data_df.columns
        data = []
        for _, row in data_df.iterrows():
            label = ['0' for _ in range(cfg.num_classes)]
            label.append('fold' + str(row.fold) + '/' + row.slice_file_name)
            label[label_map[row['class']]] = '1'
            data.append(label)
        return data

    all_data = pd.read_csv(info_file, header=0)

    ALL = get_data(all_data)
    Train, Val = train_test_split(ALL, train_size=0.8, random_state=42)

    np.savetxt("{}/music_tagging_train_urbansound8k.csv".format(info_path),
               np.array(Train),
               fmt='%s',
               delimiter=',')
    np.savetxt("{}/music_tagging_val_urbansound8k.csv".format(info_path),
               np.array(Val),
               fmt='%s',
               delimiter=',')


def generator_md(info_name, file_path, num_classes):
    """
    generate numpy array from features of all audio clips

    Args:
        audio_path (str): path to the information file.
        file_path (str): path to the npy files.

    Returns:
        2 numpy array.
        data: shape(1, 96, 1366) np.float64
        label: shape(50) np.int32
    """
    df = pd.read_csv(os.path.abspath(info_name), header=None)
    df.columns = [str(i) for i in range(num_classes)] + ["mp3_path"]
    data = []
    label = []
    for i in tqdm(range(len(df)), desc='loading data md'):
        try:
            data.append(
                np.load(os.path.join(file_path, df.mp3_path.values[i][:-4] + '.npy')).reshape(1, 96, 1366)
            )
            label.append(np.array(df[df.columns[:-1]][i:i + 1])[0])
        except FileNotFoundError:
            print(f"Exception occurred in generator_md.\t file path: {os.path.join(file_path, df.mp3_path.values[i][:-4] + '.npy')}")

    return np.array(data, np.float32), np.array(label, dtype=np.int32)


def convert_to_mindrecord(info_name, file_path, store_path, mr_name,
                          num_classes):
    """ convert dataset to mindrecord """
    num_shard = 4
    data, label = generator_md(info_name, file_path, num_classes)
    schema_json = {
        "id": {
            "type": "int32"
        },
        "feature": {
            "type": "float32",
            "shape": [1, 96, 1366]
        },
        "label": {
            "type": "int32",
            "shape": [num_classes]
        }
    }
    os.makedirs(store_path, exist_ok=True)
    writer = FileWriter(
        os.path.join(store_path, '{}.mindrecord'.format(mr_name)), num_shard, overwrite=True)
    datax = get_data(data, label)
    writer.add_schema(schema_json, "music_tagger_schema")
    writer.add_index(["id"])
    writer.write_raw_data(datax)

    writer.commit()


if __name__ == "__main__":

    cfg.info_path = cfg.info_path.replace('MagnaTagATune', "urbansound8k")
    cfg.audio_path = cfg.audio_path.replace('MagnaTagATune/mp3', "urbansound8k")
    cfg.npy_path = cfg.npy_path.replace('MagnaTagATune', "urbansound8k")
    cfg.mr_path = cfg.mr_path.replace('MagnaTagATune', "urbansound8k")
    cfg.num_classes = 10

    if cfg.get_npy:
        GetLabel_UrbanSound8K(cfg.info_path, cfg)
        dirname = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
        print(f"audio path : {cfg.audio_path}")
        for d in tqdm(dirname):  # d 表示下一级子目录
            if not os.path.isdir("{}/{}".format(cfg.audio_path, d)):
                continue
            if ".npy" in "{}/{}".format(cfg.audio_path, d):
                continue
            file_name = os.listdir("{}/{}".format(cfg.audio_path, d))
            if not os.path.isdir("{}/{}".format(cfg.npy_path, d)):
                os.makedirs("{}/{}".format(cfg.npy_path, d), exist_ok=True)
            for i, f in enumerate(tqdm(file_name)):
                compute_melgram("{}/{}/{}".format(cfg.audio_path, d, f),
                                "{}/{}/".format(cfg.npy_path, d), f)

    if cfg.get_mindrecord:
        context.set_context(device_target='Ascend', mode=context.GRAPH_MODE, device_id=7)

        for cmn in cfg.mr_name:
            if cmn in ['train', 'val']:
                convert_to_mindrecord(os.path.join(cfg.info_path, 'music_tagging_{}_urbansound8k.csv'.format(cmn)),
                                      cfg.npy_path, cfg.mr_path, cmn,
                                      cfg.num_classes)

    # only_read = True
    # mr_name = 'train'
    # if only_read:
    #     file_path = os.path.join(cfg.mr_path, '{}.mindrecord0'.format(mr_name))
    #     reader= FileReader(file_path, 4)
    #     schema = reader.schema()
    #     print(schema)
    #     length = reader.len()
    #     print(f'The number of samples: {length}')
    #     reader.close()
