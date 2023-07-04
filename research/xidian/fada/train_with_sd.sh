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

# train on source data
# python train_src.py -cfg configs/deeplabv2_r101_src.yaml --device_target Ascend OUTPUT_DIR results/src_r101_try_d6/

# resume_f ./results/src_r101_try/model_iter020000_feature_extractor.ckpt resume_c ./results/src_r101_try/model_iter020000_classifier.ckpt

# train with fine-grained adversarial alignment
# nohup python train_adv.py -cfg configs/deeplabv2_r101_adv.yaml --device_target Ascend OUTPUT_DIR results/adv_test_d19 resume_f ./results/src_r101_try_d6/model_iter017000_feature_extractor.ckpt resume_c ./results/src_r101_try_d6/model_iter017000_classifier.ckpt > ./log/train_adv_d19.out 2>&1 &

# generate pseudo labels for self distillation
# python test.py -cfg configs/deeplabv2_r101_adv.yaml --saveres resume_f ./results/adv_test_d19/model_iter026400_feature_extractor.ckpt resume_c ./results/adv_test_d19/model_iter026400_classifier.ckpt OUTPUT_DIR datasets/cityscapes/soft_labels_by_advd19_d2 DATASETS.TEST cityscapes_train
# train with self distillation
nohup python train_self_distill.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml --device_target Ascend OUTPUT_DIR results/sd_test_new_advd19 > ./log/train_dstill_advd19.out 2>&1 &

# python test.py -cfg configs/deeplabv2_r101_src.yaml resume_f ./results/src_r101_old/model_iter002000_feature_extractor.ckpt resume_c ./results/src_r101_old/model_iter002000_classifier.ckpt
