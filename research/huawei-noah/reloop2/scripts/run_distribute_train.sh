echo "Please run the script as: "
echo "bash scripts/run_distribute_train.sh DEVICE_NUM DATASET_PATH RANK_TABLE_FILE"
echo "for example: bash scripts/run_distribute_train.sh 8 /dataset_path /rank_table_8p.json"
echo "After running the script, the network runs in the background, The log will be generated in logx/output.log"


export RANK_SIZE=$1
DATA_URL=$2
export RANK_TABLE_FILE=$3

for ((i=0; i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf log$i
    mkdir ./log$i
    cp *.py ./log$i
    cp *.yaml ./log$i
    cp -r src ./log$i
    cp op_precision.ini ./log$i
    cd ./log$i || exit
    echo "start training for rank $i, device $DEVICE_ID"
    env > env.log
    python -u train.py \
    --dataset_path=$DATA_URL \
    --ckpt_path="./" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --do_eval=True > output.log 2>&1 &
    cd ../
done
