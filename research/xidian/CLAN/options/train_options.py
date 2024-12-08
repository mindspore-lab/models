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

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/media/data2/xidian/data/GTA5/'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
# INPUT_SIZE = '256,256'
# DATA_DIRECTORY_TARGET = '/data/zd/data/cityscape/cityscapes/Cityscapes'
DATA_DIRECTORY_TARGET = '/media/data2/xidian/data/Cityscapes/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DEVKIT_DIR = './dataset/cityscapes_list'
INPUT_SIZE_TARGET = '1024,512'
# INPUT_SIZE_TARGET = '256,256'

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234


#RESTORE_FROM = './model/DeepLab_resnet_pretrained_init-f81d91e8(1).ckpt'
RESTORE_FROM = '/media/data3/hy/CLAN/model/pretrained.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './checkpoint/'
SAVE_RESULT_DIR = './result/cityscapes'

WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4
LAMBDA_WEIGHT = 0.01
LAMBDA_ADV = 0.001
LAMBDA_LOCAL = 40
PREHEAT_STEPS = int(NUM_STEPS_STOP / 20)
Epsilon = 0.4
GAN = 'Vanilla'
TARGET = 'cityscapes'
SET = 'train'

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
        parser.add_argument("--model", type=str, default=MODEL,
                            help="available options : DeepLab")
        parser.add_argument("--target", type=str, default=TARGET,
                            help="available options : cityscapes")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
        parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                            help="Accumulate gradients for ITER_SIZE iterations.")
        parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                            help="number of workers for multithread dataloading.")
        parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                            help="Path to the directory containing the source dataset.")
        parser.add_argument('--devkit_dir', type=str, default=DEVKIT_DIR,
                            help='base directory of cityscapes.')
        parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                            help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")
        parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                            help="Comma-separated string with height and width of source images.")
        parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                            help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                            help="Path to the file listing the images in the target dataset.")
        parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                            help="Comma-separated string with height and width of target images.")
        parser.add_argument("--is-training", action="store_true",
                            help="Whether to updates the running means and variances during the training.")
        parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                            help="Base learning rate for discriminator.")
        parser.add_argument("--lambda-weight", type=float, default=LAMBDA_WEIGHT,
                            help="lambda-weight.")
        parser.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV,
                            help="lambda_adv for adversarial training.")
        parser.add_argument("--lambda-local", type=float, default=LAMBDA_LOCAL,
                            help="lambda-local for adversarial training.")
        parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--not-restore-last", action="store_true",
                            help="Whether to not restore last (FC) layers.")
        parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                            help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                            help="Number of training steps for early stopping.")
        parser.add_argument("--preheat-steps", type=int, default=PREHEAT_STEPS,
                            help="Number of training steps for preheat steps.")
        parser.add_argument("--epsilon", type=float, default=Epsilon,
                            help="Parameter to compute the loss.")
        parser.add_argument("--power", type=float, default=POWER,
                            help="Decay parameter to compute the learning rate.")
        parser.add_argument("--random-mirror", action="store_true",
                            help="Whether to randomly mirror the inputs during the training.")
        parser.add_argument("--random-scale", action="store_true",
                            help="Whether to randomly scale the inputs during the training.")
        parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                            help="Random seed to have reproducible results.")
        parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                            help="Where restore model parameters from.")
        parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                            help="How many images to save.")
        parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                            help="Save summaries and checkpoint every often.")
        parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                            help="Where to save snapshots of the model.")
        parser.add_argument('--save_result_path', type=str, default=SAVE_RESULT_DIR,
                            help='保存中间分割结果的路径。')
        parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--device", type=str, default='Ascend', choices=['cpu', 'gpu', 'ascend'],
                            help="choose device. ")
        parser.add_argument("--set", type=str, default=SET,
                            help="choose adaptation set.")
        parser.add_argument("--gan", type=str, default=GAN,
                            help="choose the GAN objective.")
        parser.add_argument('--debug', action='store_true', default=False,
                            help='whether use debug mode.')
        parser.add_argument('--continue-train', type=str, default='/media/data3/hy/CLAN/checkpoint/2024-09-14-08-04-48/GTA5_54000.ckpt',
                            help='whether use continue train.')
        #parser.add_argument('--continue-train', type=str, default=False, help='continue training from saved model')
        parser.add_argument('--not-val', action='store_false', default=True,
                            help='whether processing validation during the  training.')

        
        return parser.parse_args()
        