#!/bin/bash

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

WORKER_NUM=$1
DATASET=$(get_real_path $2)
CONFIG_PATH=$(get_real_path $3)
mkdir "./output"

msrun --bind_core=True --worker_num=$WORKER_NUM --local_worker_num=$WORKER_NUM --master_port=8118 \
      --log_dir=msrun_log --join=True --cluster_time_out=300 \
      train.py --run_distribute=True --data_dir=$DATASET --config=$CONFIG_PATH --output_path='./output' > log.txt 2>&1 &
