rm -rf device
mkdir device
echo "start training"
RANK_SIZE=8
for((i=0;i<${RANK_SIZE};i++));
do
    export MS_WORKER_NUM=${RANK_SIZE}  # 设置集群中Worker进程数量为8
    export MS_SCHED_HOST=127.0.0.1     # 设置Scheduler IP地址为本地环路地址
    export MS_SCHED_PORT=8118          # 设置Scheduler端口
    export MS_ROLE=MS_WORKER           # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i               # 设置进程id，可选
    python multi_infer.py \
        --encoder ./bge-large-zh-v1.5 \
        --fp16 \
        --add_instruction \
    > device/worker_$i.log 2>&1 &
done

export MS_WORKER_NUM=${RANK_SIZE}   # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1      # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=8118           # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
python multi_infer.py > device/scheduler.log 2>&1 &