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

CUDA_VISIBLE_DEVICES=$1

# train on source data
python train_src.py --device_target CPU -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/
# train with fine-grained adversarial alignment
#python train_adv.py -cfg configs/deeplabv2_r101_adv.yaml OUTPUT_DIR results/adv_test resume results/src_r101_try/model_iter020000.pth
# generate pseudo labels for self distillation
#python test.py -cfg configs/deeplabv2_r101_adv.yaml --saveres resume results/adv_test/model_iter040000.pth OUTPUT_DIR datasets/cityscapes/soft_labels DATASETS.TEST cityscapes_train
# train with self distillation
#python train_self_distill.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml OUTPUT_DIR results/sd_test