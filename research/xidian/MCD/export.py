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

import mindspore as ms
import numpy as np
from mindspore import Tensor
from mindspore import export
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import set_context
from mindspore.nn import Cell
from mindspore.ops import ArgMaxWithValue

from src.svhn2mnist import Net


class InferenceNet(Cell):
    def __init__(self, net):
        super(InferenceNet, self).__init__()
        self.net = net
        self.max = ArgMaxWithValue(axis=-1)

    def construct(self, img):
        _, out, _ = self.net(img)
        pred = self.max(out)[0]
        return pred


def main():
    """Export mindir for 310 inference."""
    parser = argparse.ArgumentParser("MatchNet exporting.")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID. ")
    parser.add_argument("--checkpoint_file", type=str, help="Checkpoint file path. ")
    parser.add_argument("--file_name", type=str, help="Output file name. ")
    parser.add_argument("--file_format", type=str, default="MINDIR",
                        choices=["AIR", "MINDIR"], help="Output file format. ")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["Ascend", "GPU", "CPU"], help="Device target.")

    args = parser.parse_args()

    set_context(mode=ms.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "Ascend":
        set_context(device_id=args.device_id)

    net = Net()
    params_dict = load_checkpoint(args.checkpoint_file)
    load_param_into_net(net, params_dict, strict_load=True)
    inference_net = InferenceNet(net)
    inference_net.set_train(False)

    input_data = Tensor(np.zeros([1, 3, 32, 32], dtype=np.float32))
    export(InferenceNet, input_data, file_name=args.file_name, file_format=args.file_format)


if __name__ == "__main__":
    main()
