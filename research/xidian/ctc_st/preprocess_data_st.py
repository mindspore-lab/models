# Copyright 2021 Huawei Technologies Co., Ltd
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

"""preprocess data and convert to mindrecord"""

import os
import string
import logging
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
from mindspore.mindrecord import FileWriter
from src.model_utils.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CHARSET = set(string.ascii_lowercase + ' ')
WORD_DIC = {v: k for k, v in enumerate(string.ascii_lowercase + ' ')}

def find_text_for_audio(file_name, txt_file_path, delimiter='\t'):
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(delimiter)
            if len(parts) >= 2:  
                audio_file, text = parts[0], delimiter.join(parts[1:]) 
                if audio_file == file_name:
                    return text



def read_timit_txt(f):
    '''read text label'''
    f=f.strip("'")
    txt = '/media/xidian/tf/ctcmodel/data_st/text.txt'
    print(f)
    line = find_text_for_audio(f,txt)
    print(line)
    line = ' '.join(line)
    line = line.replace('.', '').lower()
    line = filter(lambda c: c in CHARSET, line)
    ret = []
    for c in line:
        ret.append(WORD_DIC[c])
    return np.asarray(ret)

def diff_feature(feat, nd=1):
    '''differentiate feature'''
    diff = feat[1:] - feat[:-1]
    feat = feat[1:]
    if nd == 1:
        return np.concatenate((feat, diff), axis=1)
    d2 = diff[1:] - diff[:-1]
    return np.concatenate((feat[1:], diff[1:], d2), axis=1)


def read_files(root_path):
    '''read files'''
    files = os.walk(root_path)
    filelists = []
    for filepath, _, filenames in files:
        for filename in filenames:
            filelists.append(os.path.join(filepath, filename))
    return filelists


def get_feature(f):
    '''extract feature'''
    fs, signal = wavfile.read(f)
    signal = signal.astype('float32')
    feat = mfcc(signal=signal, samplerate=fs, winlen=0.01, winstep=0.005, numcep=13, nfilt=26, lowfreq=0, highfreq=6000,
                preemph=0.95, appendEnergy=False)
    feat = diff_feature(feat, nd=2)
    return feat


class TIMIT_PARSER():
    """
    Parse the dataset,extract the feature by mfcc,convert to mindrecord
    """

    def __init__(self, dirname, output_path):
        self.dirname = dirname
        assert os.path.isdir(dirname), dirname
        self.filelists = read_files(self.dirname)
                          
        self.output_path = output_path

    def getdatas(self):
        '''get data'''
        data = []
        for f in self.filelists:
            feat = get_feature(f)
            label = read_timit_txt(f[-24:])
            data.append([feat, label])
        return data

    def convert_to_mindrecord(self):
        '''convert to mindrecord'''
        schema_json = {"id": {"type": "int32"},
                       "feature": {"type": "float32", "shape": [-1, 39]},
                       "masks": {"type": "float32", "shape": [-1, 256]},
                       "label": {"type": "int32", "shape": [-1]},
                       "seq_len": {"type": "int32"},
                       }
        data_list = []
        logger.info("write into mindrecord,plaese wait")
        pair = self.getdatas()
        for i, data in enumerate(pair):
            feature = data[0]
            label = data[1]
            if feature.shape[0] > config.max_sequence_length:
                feature = feature[:config.max_sequence_length]
            feature_padding = np.zeros((config.max_sequence_length, feature.shape[1]), dtype=np.float32)
            feature_padding[:feature.shape[0], :] = feature
            masks = np.zeros((config.max_sequence_length, 2 * config.hidden_size), dtype=np.float32)
            masks[:feature.shape[0], :] = 1
            if label.shape[0] > config.max_label_length:
                label = label[:config.max_label_length]
            label_padding = np.full(config.max_label_length, 61, dtype=np.int32)
            label_padding[:label.shape[0]] = label
            data_json = {"id": i,
                         "feature": feature_padding.reshape(-1, config.feature_dim),
                         "masks": masks.reshape(-1, 2 * config.hidden_size),
                         "label": label_padding.reshape(-1),
                         "seq_len": feature.shape[0],
                         }
            data_list.append(data_json)
        writer = FileWriter(self.output_path, shard_num=4)
        writer.add_schema(schema_json, "nlp_schema")
        writer.add_index(["id"])
        writer.write_raw_data(data_list)
        writer.commit()
        logger.info("writing into record suceesfully")


if __name__ == '__main__':
    if not os.path.exists(config.dataset_dir):
        os.makedirs(config.dataset_dir)
    logger.info("Preparing train dataset:")
    train_path = os.path.join(config.dataset_dir, config.train_name)
    parser = TIMIT_PARSER(config.train_dir, train_path)
    parser.convert_to_mindrecord()
    logger.info("Preparing test dataset:")
    test_path = os.path.join(config.dataset_dir, config.test_name)
    parser = TIMIT_PARSER(config.test_dir, test_path)
    parser.convert_to_mindrecord()
