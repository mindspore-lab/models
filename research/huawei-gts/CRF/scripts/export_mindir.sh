#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash export_ascend.sh DEVICE_ID TRAIN_DATA_PATH MODEL_FILE"
echo "for example: bash export_mindir.sh 0 ../conll2003 lstm_crf.mindir"
echo "=============================================================================================================="

DEVICE_ID=$1
TRAIN_DATA_FOLDER=$2
MODEL_FILE=$3

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"


python "${BASE_PATH}/../src/export.py"  \
    --config_path=$CONFIG_FILE \
    --device_id=${DEVICE_ID}\
    --device_target="Ascend" \
    --model_format="MINDIR" \
    --data_path=${TRAIN_DATA_FOLDER}\
    --model_path=${MODEL_FILE}
#    --model_path=${CKPT_FILE}  > log_export.txt 2>&1 &
