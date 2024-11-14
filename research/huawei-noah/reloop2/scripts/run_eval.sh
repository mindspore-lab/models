echo "Please run the script as: "
echo "bash scripts/run_eval.sh DEVICE_ID DEVICE_TARGET DATASET_PATH CHECKPOINT_PATH"
echo "for example: bash scripts/run_eval.sh 0 GPU /dataset_path /checkpoint_path"
echo "After running the script, the network runs in the background, The log will be generated in ms_log/eval_output.log"

export DEVICE_ID=$1
DEVICE_TARGET=$2
DATA_URL=$3
CHECKPOINT_PATH=$4

mkdir -p ms_log
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0

python -u eval.py \
    --dataset_path=$DATA_URL \
    --checkpoint_path=$CHECKPOINT_PATH \
    --device_target=$DEVICE_TARGET > ms_log/eval_output.log 2>&1 &
