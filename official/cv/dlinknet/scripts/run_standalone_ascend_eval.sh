#!/bin/bash

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# != 5 ] && [ $# != 6 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash scripts/run_standalone_ascend_eval.sh [DATASET] [LABEL_PATH] [CHECKPOINT] [PREDICT_PATH] [CONFIG_PATH] [DEVICE_ID](option, default is 0)"
    echo "for example: bash run_standalone_ascend_eval.sh /path/to/data/ /path/to/label/ /path/to/checkpoint/ /path/to/predict/ /path/to/config/ 0"
    echo "=============================================================================================================="
    exit 1
fi
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export DEVICE_ID=0
if [ $# != 5 ]
then
  export DEVICE_ID=$6
fi
rm -rf "$4"
mkdir "$4"
DATASET=$(get_real_path $1)
LABEL_PATH=$(get_real_path $2)
CHECKPOINT=$(get_real_path $3)
PREDICT_PATH=$(get_real_path $4)
CONFIG_PATH=$(get_real_path $5)
echo "========== start run evaluation ==========="
echo "please get log at eval.log"
python ${PROJECT_DIR}/../eval.py --data_dir=$DATASET --label_path=$LABEL_PATH --trained_ckpt=$CHECKPOINT --predict_path=$PREDICT_PATH --config=$CONFIG_PATH  > eval.log 2>&1 &
