import sys
from pathlib import Path
import numpy as np
import cv2
from munch import DefaultMunch

import mindspore_lite as mslite

# sys.path.append('/home/linzheyuan/wukong-huahua/LLVT')
from utils.task_metaclass import Task


def test_rrdb_om_srx4(input_data,  model_file):
    print("start: ==========================================")
    cfg = {
        "task_name": "super_resolution_x4",
        "task_type": "super_resolution",
        "data_io": "BasicVSR_MindSpore",
        "backend": "ascend",
        "model_file": model_file,
        "once_process_frames": 1,
        "up_scale": 4,
        "device_id": 0,
    }
    task = Task(DefaultMunch.fromDict(cfg))
    # input_data = cv2.imread(img_file)
    h, w, c = input_data.shape
    output_data = task.run([input_data,], **{"height": h, "width": w})
    # print(output_data[0])
    # print(output_file)
    # ret = cv2.imwrite(output_file, output_data[0])
    return output_data[0]

# test_rrdb_om_srx4(
#         "/home/linzheyuan/wukong-huahua/LLVT/output/512x512.png",
#         "/home/linzheyuan/wukong-huahua/LLVT/output/sample.png",
#         "/home/linzheyuan/wukong-huahua/LLVT/rrd.om"
#         )
