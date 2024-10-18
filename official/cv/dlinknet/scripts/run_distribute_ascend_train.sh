#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 3 ]
then
    echo "=============================================================================================================="
    echo "Usage: bash scripts/run_distribute_ascend_train.sh [WORKER_NUM] [DATASET] [CONFIG_PATH]"
    echo "Please run the script as: "
    echo "bash scripts/run_distribute_ascend_train.sh [WORKER_NUM] [DATASET] [CONFIG_PATH]"
    echo "for example: bash run_distribute_ascend_train.sh 8 /absolute/path/to/data /absolute/path/to/config"
    echo "=============================================================================================================="
    exit 1
fi
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export HCCL_CONNECT_TIMEOUT=600

WORKER_NUM=$(get_real_path $1)
DATASET=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $3)
mkdir "./output"

msrun --bind_core=True --worker_num=$WORKER_NUM --local_worker_num=$WORKER_NUM --master_port=8118 \
      --log_dir=msrun_log --join=True --cluster_time_out=300 \
      train.py --run_distribute=True --data_path=$DATASET --config_path=$CONFIG_PATH --output_path='./output'
