#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh RANK_SIZE TASK MODEL DATASET"
echo "For example: bash run.sh 8 traffic_state_pred STResNet NYCBike20140409"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
#DATA_PATH=$1
#export DATA_PATH=${DATA_PATH}
RANK_SIZE=$1
export HCCL_CONNECT_TIMEOUT=6000

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    python hccl_tool.py --device_num "[0,8)"
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    python hccl_tool.py --device_num "[0,2)"
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_1pcs()
{
#    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=1
}


test_dist_${RANK_SIZE}pcs

cd `dirname $0`
dir=`pwd`
cd $dir

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp -a ./config ./data ./evaluator ./executor ./kernel_meta ./log ./model ./pipeline ./rank_0 ./raw_data ./utils ./test_pipeline.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./test_pipeline.py --task $2 --model $3 --dataset $4 > train.log$i 2>&1 &
    cd ../
done

rm -rf device0
mkdir device0
cp -a ./config ./data ./evaluator ./executor ./kernel_meta ./log ./model ./pipeline ./rank_0 ./raw_data ./utils ./test_pipeline.py  ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python ./test_pipeline.py --task $2 --model $3 --dataset $4
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
