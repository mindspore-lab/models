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
"""Export checkpoint into mindir or air for inference."""
import argparse
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import Softmax
from mindspore import Tensor
from mindspore import set_context
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import export

from src.MatchNet import MatchNet


class InferModel(nn.Cell):
    """Add resize and exp behind HRNet."""
    def __init__(self):
        super(InferModel, self).__init__()
        self.net = MatchNet()
        self.softmax = Softmax()

    def construct(self, img1, img2):
        """Model construction."""
        logits = self.softmax(self.net((img1, img2)))
        pred_scores = logits[:, 1]
        return pred_scores


def main():
    """Export mindir for 310 inference."""
    parser = argparse.ArgumentParser("HRNet Semantic Segmentation exporting.")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID. ")
    parser.add_argument("--checkpoint_file", type=str, help="Checkpoint file path. ")
    parser.add_argument("--file_name", type=str, help="Output file name. ")
    parser.add_argument("--file_format", type=str, default="MINDIR",
                        choices=["AIR", "MINDIR"], help="Output file format. ")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend", "GPU", "CPU"], help="Device target.")
    parser.add_argument("--dataset", type=str, default="cityscapes")

    args = parser.parse_args()

    set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        set_context(device_id=args.device_id)

    net = MatchNet()
    net.set_train(False)

    params_dict = load_checkpoint(args.checkpoint_file)
    load_param_into_net(net, params_dict, strict_load=True)

    input_data = Tensor(np.zeros([1, 1, 64, 64], dtype=np.float32))
    export(net, input_data, file_name=args.file_name, file_format=args.file_format)


if __name__ == "__main__":
    main()
