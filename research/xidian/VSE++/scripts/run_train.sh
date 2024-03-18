#!/bin/bash
nohup python train.py --device_id 0  > train_Ascend.log 2>&1 &
