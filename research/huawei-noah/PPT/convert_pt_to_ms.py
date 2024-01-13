# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Convert PeTorch model weights to Mindspore checkpoint."""

import argparse

import mindspore as ms
from loguru import logger
import torch
from src.model.factory import create_model, create_teacher_model

pytorch_checkpoints_urls = {
    'deit_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/'
                             'deit_base_patch16_224-b5f2ef4d.pth',
    'deit_base_distilled_patch16_224':
        'https://dl.fbaipublicfiles.com/deit/'
        'deit_base_distilled_patch16_224-df68dfff.pth',
    'deit_tiny_patch16_224': 'https://dl.fbaipublicfiles.com/deit/'
                             'deit_tiny_patch16_224-a1311bcf.pth',
    'deit_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/'
                              'deit_small_patch16_224-cd65a155.pth',
    'deit_tiny_distilled_patch16_224':
        'https://dl.fbaipublicfiles.com/deit/'
        'deit_tiny_distilled_patch16_224-b40b3cf7.pth',
    'deit_small_distilled_patch16_224':
        'https://dl.fbaipublicfiles.com/deit/'
        'deit_small_distilled_patch16_224-649709d9.pth',
    'deit_base_patch16_384': 'https://dl.fbaipublicfiles.com/deit/'
                             'deit_base_patch16_384-8de9b5d1.pth',
    'deit_base_distilled_patch16_384':
        'https://dl.fbaipublicfiles.com/deit/'
        'deit_base_distilled_patch16_384-d0272ac0.pth',
    'regnety_160': 'https://github.com/rwightman/pytorch-image-models/'
                   'releases/download/v0.1-regnet/regnety_160-d64013cd.pth'

}


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="MindSpore ConvMixer convert weights PyTorch -> Mindspore"
    )
    parser.add_argument('--model', default='deit_base_patch16_224', type=str,
                        choices=[
                            'deit_base_patch16_224',
                            'deit_base_distilled_patch16_224',
                            'deit_tiny_patch16_224',
                            'deit_small_patch16_224',
                            'deit_tiny_distilled_patch16_224',
                            'deit_small_distilled_patch16_224',
                            'deit_base_patch16_384',
                            'deit_base_distilled_patch16_384',
                            'regnety_160'
                        ],
                        help='Name of model to move weights')
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument(
        "--pt_state_dict",
        type=str,
        help="Path to PyTorch checkpoint file."
    )
    parser.add_argument(
        "--ms_checkpoint",
        type=str,
        help="Path to Mindspore checkpoint file."
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO"],
        default='DEBUG',
        help="Logging level."
    )

    return parser.parse_args()


def update_net_params(ms_net, pt_state_dict):
    """Updates parameters in mindspore model by pytorch state dict"""
    ms_name_map = {
        name
        .replace('network.', '')
        .replace('s.', 's')
        .replace('b.', 'b')
        .replace('.beta', '.bias')
        .replace('.moving_variance', '.running_var')
        .replace('.moving_mean', '.running_mean')
        .replace('.gamma', '.weight'): name
        for name, param in ms_net.parameters_and_names()
    }  # pt -> ms

    names_ms = []
    names_pt = []
    ms_net_dict = {n: p for n, p in ms_net.parameters_and_names()}

    for name_pt, param_pt in pt_state_dict.items():
        if name_pt not in ms_name_map.keys():
            logger.debug(f'Not found corresponding name for: {name_pt}')
            continue
        name_ms = ms_name_map[name_pt]
        names_ms.append(name_ms)
        names_pt.append(name_pt)
        ms_net_dict[name_ms].set_data(
            ms.Tensor(param_pt.data.cpu().numpy())
        )
    return ms_net


def main():
    args = parse_args()

    logger.level(args.log_level)

    if args.model == 'regnety_160':
        net = create_teacher_model(args.model)

        pt_state_dict = torch.hub.load_state_dict_from_url(
            url=pytorch_checkpoints_urls[args.model],
            map_location="cpu", check_hash=True
        )
    else:
        net = create_model(args.model, args.num_classes)
        if not args.pt_state_dict:
            pt_state_dict = torch.hub.load_state_dict_from_url(
                url=pytorch_checkpoints_urls[args.model],
                map_location="cpu", check_hash=True
            )['model']
        else:
            pt_state_dict = torch.load(args.pt_state_dict)['model']

    logger.info('Created MS model and loaded PT state dict.')
    update_net_params(net, pt_state_dict)
    logger.info('Updated weights.')
    ms.save_checkpoint(net, args.ms_checkpoint)
    logger.info('Complete.')


if __name__ == '__main__':
    main()
