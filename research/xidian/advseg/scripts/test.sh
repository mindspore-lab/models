#!/bin/bash
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

# export DEVICE_NUM=$1
export DEVICE_NUM=1
export RANK_SIZE=${DEVICE_NUM}

DATA_PATH="/home/ma-user/work/datasets"
#DATA_PATH=$2
GTA5_PATH="${DATA_PATH}/gtav"
CITYSCAPES_PATH="${DATA_PATH}/Cityscapes"


LOG_PATH="./log"

EXEC_PATH="$(pwd)"
echo "the work path : $EXEC_PATH"
CONFIG_PATH="${EXEC_PATH}/advnet_config.yaml"
#CONFIG_PATH="./advnet_config.yaml"
echo "config path is : ${CONFIG_PATH}"
# python train.py --data_dir ${GTA5_PATH} \
#                         --data_dir_target ${CITYSCAPES_PATH} \
#                         --config_path ${CONFIG_PATH}
python eval.py --data_dir ${GTA5_PATH} \
                        --data_dir_target ${CITYSCAPES_PATH} \
                        --config_path ${CONFIG_PATH}
                        --restrore_from ./checkpoint/best_41.61.ckpt