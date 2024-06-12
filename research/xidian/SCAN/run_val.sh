

epochs=20  # 训练epoch
result_path="logs/"             # 保存log和权重的总路径
log_path="${result_path}log"
weight_path="${result_path}checkpoint"
datapath="./data/"
device_id=0

# 进行验证，保留验证集最好的模型
nohup python  -u val.py --data_path  $datapath   --model_name $weight_path  --num_epochs $epochs  --device_target Ascend  --device_id $device_id  > eval.log 2>&1 &