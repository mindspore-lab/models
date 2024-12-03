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


def GetLabel_IRMAS(audio_path):
    """
    separate dataset into training set and validation set

    Args:
        audio_path (str): path to the information file.

    """

    map_dict1={'cel':0, 'cla':1, 'flu':2, 'gac':3, 'gel':4, 'org':5, 'pia':6, 'sax':7, 'tru':8, 'vio':9, 'voi':10}
    map_dict2={'dru':11, 'nod':12}
    map_dict3={'cla':13, 'cou_fol':14, 'jaz_blu':15, 'lat_sou':16, 'pop_roc':17}


    dict1,dict2,dict3 = dict({k:0 for k in map_dict1.keys()}),dict(),dict()

    set1, set2, set3 = set(), set(), set()

    all_data = []
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith('.wav'):

                label = ['0' for _ in range(len(map_dict1)+len(map_dict2)+len(map_dict3))]
                label.append(os.path.join(root, file).replace(audio_path+'/',''))

                filename_keywords = extract_keywords_from_filename(file)
                # print(f"extract from {file},\t len:{len(filename_keywords)},\t result : {filename_keywords}")
                assert len(filename_keywords) == 2 or len(filename_keywords) == 3

                label[map_dict1[filename_keywords[0]]]='1'
                label[map_dict3[filename_keywords[-1]]]='1'
                set1.add(filename_keywords[0])
                set3.add(filename_keywords[-1])

                if len(filename_keywords) == 3 and filename_keywords[1] != '---':
                    label[map_dict2[filename_keywords[1]]] = '1'
                    set2.add(filename_keywords[1])
                # print(f"sample:{label}")
                all_data.append(label)


    # print(f"set1 : {set1},\t set2 : {set2},\t set3 : {set3}")

    Train,Val = train_test_split(all_data,train_size=0.8,random_state=42)
    # raise None
    # Train = []
    # Val = []

    np.savetxt("{}/music_tagging_train_IRMAS.csv".format(audio_path),
               np.array(Train),
               fmt='%s',
               delimiter=',')
    np.savetxt("{}/music_tagging_val_IRMAS.csv".format(audio_path),
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
            print("Exception occurred in generator_md.")

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

    # print(len(writer))
    # print(f'data shape:{}')
    writer.commit()


if __name__ == "__main__":

    cfg.audio_path = cfg.audio_path.replace('MagnaTagATune/mp3', "IRMAS/train")
    cfg.npy_path = cfg.npy_path.replace('MagnaTagATune', "IRMAS")
    cfg.mr_path = cfg.mr_path.replace('MagnaTagATune', "IRMAS")
    cfg.num_classes=18

    if cfg.get_npy:
        GetLabel_IRMAS(cfg.audio_path)
        dirname = os.listdir(cfg.audio_path)
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
        context.set_context(device_target='Ascend', mode=context.GRAPH_MODE, device_id=get_device_id())

        for cmn in cfg.mr_name:
            if cmn in ['train', 'val']:
                convert_to_mindrecord(os.path.join(cfg.audio_path, 'music_tagging_{}_IRMAS.csv'.format(cmn)),
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
