result_path="logs/"             # 保存log和权重的总路径
log_path="${result_path}log"
weight_path="${result_path}checkpoint"
data_path="data/"
device_id=0

nohup python -u test.py --data_path $data_path   --model_name $weight_path   --device_target Ascend  --device_id $device_id   > test.log 2>&1 &