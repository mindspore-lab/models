echo "Please run the script as: "
echo "bash scripts/run_standalone_train.sh DEVICE_ID/CUDA_VISIBLE_DEVICES DEVICE_TARGET DATASET_PATH"
echo "for example: bash scripts/run_standalone_train.sh 0 GPU /dataset_path"
echo "After running the script, the network runs in the background, The log will be generated in ms_log/output.log"

DEVICE_TARGET=$2

if [ "$DEVICE_TARGET" = "GPU" ]
then
  export CUDA_VISIBLE_DEVICES=$1
fi

if [ "$DEVICE_TARGET" = "Ascend" ];
then
  export DEVICE_ID=$1
fi

DATA_URL=$3

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python -u train.py \
    --dataset_path=$DATA_URL \
    --ckpt_path="checkpoint" \
    --eval_file_name='auc.log' \
    --loss_file_name='loss.log' \
    --device_target=$DEVICE_TARGET \
    --do_eval=True > ms_log/output.log 2>&1 &
