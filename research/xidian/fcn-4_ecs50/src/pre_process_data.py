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
from random import shuffle

import librosa
import numpy as np
import pandas as pd
from mindspore import context
from mindspore.mindrecord import FileWriter

from src.model_utils.config import config as cfg
from src.model_utils.device_adapter import get_device_id


def compute_melgram(audio_path, save_path='', save_npy=True):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA =29.12 # to make it 235 frame..

    try:
        src, _ = librosa.load(audio_path, sr=SR)  # whole signal
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
    if save_npy:
        save_path = save_path[:-4] + '.npy'
        np.save(save_path, ret)


def get_data(features_data, labels_data):
    data_list = []
    for i, (label, feature) in enumerate(zip(labels_data, features_data)):
        data_json = {"id": i, "feature": feature, "label": label}
        data_list.append(data_json)
    return data_list


def generator_md(info_name):
    df = pd.read_csv(info_name, header=None)
    data = []
    label = []
    for i in range(1, len(df)):
        try:
            data.append(
                np.load(os.path.dirname(info_name) + "/datas_npy/" + df[0][i].replace("wav", "npy")).reshape(1, 96, -1))
            onghot = np.zeros((1, 50), dtype=int)
            onghot[0][int(df[1][i])] = 1
            label.append(onghot)
        except FileNotFoundError:
            print("Exception occurred in generator_md.")
            print(df.mp3_path.values[i][:-4])
    return np.array(data), np.array(label, dtype=np.int32)


def convert_to_mindrecord(info_name, store_path, mr_name,
                          num_classes):
    """ convert dataset to mindrecord """
    num_shard = 1
    data, label = generator_md(info_name)
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

    writer = FileWriter(
        os.path.join(store_path, '{}.mindrecord'.format(mr_name)), num_shard)
    datax = get_data(data, label)
    writer.add_schema(schema_json, "music_tagger_schema")
    writer.add_index(["id"])
    writer.write_raw_data(datax)
    writer.commit()


if __name__ == "__main__":
    data_train = {"data": [], "label": []}
    data_val = {"data": [], "label": []}
    if cfg.get_npy:
        dirname = os.listdir(cfg.audio_path)
        shuffle(dirname)
        for i, d in enumerate(dirname):
            if i < len(dirname) * 0.8:
                data_train["data"].append(d)
                data_train["label"].append(int(os.path.basename(d).split("-")[-1].split(".")[0]))
            else:
                data_val["data"].append(d)
                data_val["label"].append(int(os.path.basename(d).split("-")[-1].split(".")[0]))
            file_name = os.path.join("{}/{}".format(cfg.audio_path, d))
            if not os.path.isdir(cfg.npy_path):
                os.mkdir(cfg.npy_path)

            compute_melgram(audio_path=file_name, save_path="{}/{}".format(cfg.npy_path, d), save_npy=True)

        pd.DataFrame(data_train).to_csv("{}/train.csv".format(cfg.data_path), index=False, decimal=',')
        pd.DataFrame(data_val).to_csv("{}/val.csv".format(cfg.data_path), index=False, decimal=',')

    if cfg.get_mindrecord:
        context.set_context(device_target='Ascend', mode=context.GRAPH_MODE, device_id=get_device_id())

        for cmn in cfg.mr_name:
            if cmn in ['train', 'val']:
                convert_to_mindrecord('./data/{}.csv'.format(cmn),
                                      cfg.mr_path, cmn,
                                      cfg.num_classes)
