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
import argparse

DATA_DIRECTORY = '/media/data2/xidian/data/Cityscapes/'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500  # Number of images in the validation set.
INPUT_SIZE = '1024,512'
OUTPUT_SIZE = '1024,2048'

SET = 'val'

MODEL = 'DeeplabMulti'

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
        parser.add_argument("--model", type=str, default=MODEL,
                            help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
        parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                            help="Path to the directory containing the Cityscapes dataset.")
        parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                            help="Path to the file listing the images in the dataset.")
        parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")
        parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--restore-from", type=str, default='/media/data3/hy/CLAN/checkpoint/2024-09-14-08-04-48/GTA5_best.ckpt',
                            help="Where restore model parameters from.")
        parser.add_argument('--input_size', type=str, default=INPUT_SIZE,
                            help='the image input crop size.')
        parser.add_argument('--output_size', type=str, default=OUTPUT_SIZE,
                            help='the image output size')
        parser.add_argument("--gpu", type=int, default=1,
                            help="choose gpu device.")
        parser.add_argument("--set", type=str, default=SET,
                            help="choose evaluation set.")
        parser.add_argument("--save_path", type=str, default=SAVE_PATH,
                            help="Path to save result.")
        parser.add_argument('--devkit-dir', default='dataset/cityscapes_list',
                            help='base directory of cityscapes')
        return parser.parse_args()