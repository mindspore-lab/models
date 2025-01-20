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


# python infer_src.py -cfg configs/deeplabv2_r101_src.yaml --device_target Ascend resume_f ./results/src_r101_try_d6/model_iter018000_feature_extractor.ckpt resume_c ./results/src_r101_try_d6/model_iter018000_classifier.ckpt 


# python infer_src.py -cfg configs/deeplabv2_r101_adv.yaml --device_target Ascend resume_f ./results/adv_test_d14/model_iter037000_feature_extractor.ckpt resume_c ./results/adv_test_d14/model_iter037000_classifier.ckpt 

# python infer_adv.py -cfg configs/deeplabv2_r101_adv.yaml --device_target Ascend resume_f ./results/adv_test_d9/model_iter035000_feature_extractor.ckpt resume_c ./results/adv_test_d9/model_iter035000_classifier.ckpt 
# resume_d ./results/adv_test/model_iter038000_discriminator.ckpt OUTPUT_DIR results/adv_test/

python infer_adv.py -cfg configs/deeplabv2_r101_adv.yaml --device_target Ascend 
# python infer_adv.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml --device_target Ascend 


# python infer_src.py -cfg configs/deeplabv2_r101_src.yaml --device_target Ascend resume_f ./results/sd_test/model_iter018000_feature_extractor.ckpt resume_c ./results/sd_test/model_iter018000_classifier.ckpt OUTPUT_DIR results/src_r101_try1/

# python test.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml resume_f ./results/sd_test_new_advd5/model_iter016000_feature_extractor.ckpt resume_c ./results/sd_test_new_advd5/model_iter016000_classifier.ckpt
