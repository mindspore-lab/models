#!/bin/bash
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


export DEVICE_ID=$1
#export DEVICE_NUM=$1
#export DEVICE_NUM=1
#export RANK_SIZE=${DEVICE_NUM}

LOG_PATH="./log"

EXEC_PATH="$(pwd)"
echo "the work path : $EXEC_PATH"

#python -u train.py --config_path "./snd_config.yaml"
nohup python train.py --config_path "./snd_config.yaml" --device_id=$DEVICE_ID > log.log 2>&1 &
#python eval.py --data_dir ${GTA5_PATH} \
#                        --data_dir_target ${CITYSCAPES_PATH} \
#                        --config_path ${CONFIG_PATH}
#                        --restrore_from ./checkpoint/best_41.61.ckpt

#mpirun --allow-run-as-root -n ${RANK_SIZE} python train.py --config_path="./snd_config.yaml" --device_num=${DEVICE_NUM}