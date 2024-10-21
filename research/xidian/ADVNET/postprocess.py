# Copyright 2023 Xidian University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''post process for 310 inference'''
import os
import numpy as np
from src.model_utils import config
from src.model_utils import compute_mIoU
from PIL import Image

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def read_predict(predict_path):
    output_size = config.output_size
    files = os.listdir(predict_path)
    result_shape = config.input_size_target
    predicts = []
    for file in files:
        full_file_path = os.path.join(predict_path, file)
        if os.path.isfile(full_file_path):
            name=file #  todo need modified
            predict = np.fromfile(full_file_path, dtype=np.float32)
            interp=ops.ResizeBilinear(size=output_size)
            predict = interp(predict)
            n, c, h, w = predict.shape
            predict = predict.reshape((c, h, w))
            predict = predict.transpose((1, 2, 0))
            predict = np.asarray(np.argmax(predict, axis=2), dtype=np.uint8)
            predict_col = colorize_mask(predict)
            predict_col.save('%s/%s_color.png' % (config.save_result, name))
            predict = Image.fromarray(predict)
            predicts.append(predict)
    return predicts

def main():
    data_dir_target=config.data_dir_target
    if os.path.exists(os.path.join(data_dir_target, 'gtFine_trainvaltest')):
        gt_path = os.path.join(data_dir_target, 'gtFine_trainvaltest', 'gtFine', 'val')
    else:
        gt_path = os.path.join(data_dir_target, 'gtFine', 'val')

    predicts = read_predict(config.predict_path)
    mIoUs, logs = compute_mIoU(gt_dir=gt_path, preds=predicts, devkit_dir=config.data_list_target)
    print(logs)
    print(mIoUs)

if __name__ == "__main__":
    main()
