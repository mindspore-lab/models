# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""TB-Net inference app."""

# This script should be run directly with 'python <script> <args>'.

import os
import io
import argparse
import json

from mindspore import load_checkpoint, load_param_into_net, set_context, PYNATIVE_MODE, GRAPH_MODE

from src.tbnet import TBNet
from src.recommend import Recommender
from src.dataset import create_dataset
from tbnet_config import TBNetConfig


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Infer TB-Net.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='infer.csv',
        help="the csv datafile inside the dataset folder (e.g. infer.csv), it contains data of only 1 user."
    )

    parser.add_argument(
        '--checkpoint_id',
        type=int,
        required=True,
        help="use which checkpoint(.ckpt) file to infer"
    )

    parser.add_argument(
        '--items',
        type=int,
        required=False,
        default=3,
        help="no. of items to be recommended"
    )

    parser.add_argument(
        '--explanations',
        type=int,
        required=False,
        default=3,
        help="no. of explanations to be shown"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='GRAPH',
        choices=['GRAPH', 'PYNATIVE'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    return parser.parse_args()


def infer_tbnet():
    """Inference process."""
    args = get_args()
    set_context(device_id=args.device_id)
    if args.run_mode == 'GRAPH':
        set_context(mode=GRAPH_MODE)
    else:
        set_context(mode=PYNATIVE_MODE)

    home = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    data_path = os.path.join(home, 'data', args.dataset, args.csv)
    id_maps_path = os.path.join(home, 'data', args.dataset, 'id_maps.json')
    ckpt_path = os.path.join(home, 'checkpoints', args.dataset, f'tbnet_epoch{args.checkpoint_id}.ckpt')

    print(f"creating TBNet from checkpoint {args.checkpoint_id}...")
    cfg = TBNetConfig(config_path)
    network = TBNet(cfg.num_items, cfg.num_references, cfg.num_relations, cfg.embedding_dim)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    network.set_train(False)

    print(f"creating dataset from {data_path}...")
    infer_ds = create_dataset(data_path, cfg.per_item_paths)
    infer_ds = infer_ds.batch(cfg.batch_size)

    print("inferring...")
    # infer and aggregate results

    with io.open(id_maps_path, mode="r", encoding="utf-8") as f:
        id_maps = json.load(f)
    recommender = Recommender(network, id_maps, args.items)

    for item, rl1, ref, rl2, hist_item, _ in infer_ds:
        # assume there is data of only one user in infer_ds and all items inside are candidates.
        recommender(item, rl1, ref, rl2, hist_item)

    # show recommendations with explanations
    suggestions = recommender.suggest()
    for suggest in suggestions:
        print("")
        print(f'Recommends item:"{suggest.item}" (score:{suggest.score}) because:')
        # show explanations
        explanation = 0
        for path in suggest.paths:
            if path.relation1 == path.relation2:
                print(f'- it shares the same {path.relation1}:"{path.reference}" with user\'s '
                      f'historical item:"{path.hist_item}".\n  (importance:{path.importance})')
            else:
                print(f'- it has {path.relation1}:"{path.reference}" while which is {path.relation2} '
                      f'of user\'s historical item:"{path.hist_item}".\n  (importance:{path.importance})')
            explanation += 1
            if explanation >= args.explanations:
                break


if __name__ == '__main__':
    infer_tbnet()
