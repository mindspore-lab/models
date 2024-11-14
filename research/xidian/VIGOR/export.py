# Copyright 2024 Xidian University
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

from src.model import VIGOR


class InferenceQueryNet(Cell):
    def __init__(self, net):
        super(InferenceQueryNet, self).__init__()
        self.net = net

    def construct(self, img):
        return self.net.grd_head(self.net.grd(img))


class InferenceRefNet(Cell):
    def __init__(self, net):
        super(InferenceRefNet, self).__init__()
        self.net = net

    def construct(self, img):
        return self.net.sat_head(self.net.sat(img))


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

    net = VIGOR()
    params_dict = load_checkpoint(args.checkpoint_file)
    load_param_into_net(net, params_dict, strict_load=True)
    infer_query_net = InferenceQueryNet(net)
    infer_ref_net = InferenceRefNet(net)
    infer_query_net.set_train(False)
    infer_ref_net.set_train(False)

    input_query_data = Tensor(np.zeros([1, 3, 640, 320], dtype=np.float32))
    input_ref_data = Tensor(np.zeros([1, 3, 640, 640], dtype=np.float32))
    export(infer_query_net, input_query_data, file_name=args.file_name+'_query', file_format=args.file_format)
    export(infer_ref_net, input_ref_data, file_name=args.file_name+'_ref', file_format=args.file_format)


if __name__ == "__main__":
    main()
