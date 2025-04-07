#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash export_ascend.sh DEVICE_ID TRAIN_DATA_FOLDER EXPORT_SUFFIX"
echo "for example: bash export_ascend.sh 0 ../conll2003 my_suffix"
echo "=============================================================================================================="

DEVICE_ID=$1
TRAIN_DATA_FOLDER=$2
EXPORT_SUFFIX=$3

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config.yaml"


python "${BASE_PATH}/../src/export.py"  \
    --config_path=$CONFIG_FILE \
    --device_id=${DEVICE_ID}\
    --data_path=${TRAIN_DATA_FOLDER}\
    --export_suffix=${EXPORT_SUFFIX}
#    --ckpt_path=${CKPT_FILE}  > log_export.txt 2>&1 &
