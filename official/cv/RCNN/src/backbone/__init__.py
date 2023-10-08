import ssl
import mindspore as ms
from mindspore import nn, ops
from mindcv import create_model
from .fpn import FPN
from .det_resnet import *
from ..utils import logger

ssl._create_default_https_context = ssl._create_unverified_context


def create_backbone(initializer, in_channels=3, pretrained=True, backbone_ckpt=""):
    """
    Creates backbone by MindCV
    Args:
        initializer (str): backbone name.
        in_channels (int): The input channels. Default: 3.
        pretrained (bool): Whether to load the pretrained model. Default: False.
        backbone_ckpt (str): The path of checkpoint files. Default: "".

    """
    if ms.get_auto_parallel_context("device_num") > 1:
        allreduce_sum = ops.AllReduce(ops.ReduceOp.SUM)
        from mindspore.communication.management import get_local_rank, get_rank

        local_rank = get_rank() % 8
        try:
            local_rank = get_local_rank()
        except:
            logger.info("Not support get_local_rank, get local_rank by get_rank() % 8")

        if local_rank == 0:
            print(f"==== create_model {local_rank} start")
            net = create_model(
                initializer, in_channels=in_channels, pretrained=pretrained, ckpt_path=backbone_ckpt
            )
            print(f"==== create_model {local_rank} done")
            r = allreduce_sum(ops.ones((1), ms.float32))
        else:
            r = allreduce_sum(ops.ones((1), ms.float32))
            print(f"==== create_model {local_rank} {r} start")
            net = create_model(
                initializer, in_channels=in_channels, pretrained=pretrained, ckpt_path=backbone_ckpt
            )
            print(f"==== create_model {local_rank} done")
    else:
        net = create_model(initializer, in_channels=in_channels, pretrained=pretrained, ckpt_path=backbone_ckpt)
    return net


def build_backbone(cfg):
    model_name = cfg.name
    network = create_backbone(model_name, pretrained=cfg.pretrained)
    if cfg.frozen_bn:
        for _, cell in network.cells_and_names():
            if isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                cell.use_batch_statistics = False
                cell.gamma.requires_grad = False
                cell.beta.requires_grad = False
    if cfg.frozen_2stage:
        network.frozen_2stage = ops.stop_gradient

    if hasattr(cfg, "fpn"):
        network = FPN(
            bottom_up=network,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            num_outs=cfg.fpn.num_outs,
            level_index=cfg.fpn.level_index,
            norm=cfg.fpn.norm,
            act=cfg.fpn.act,
            upsample_mode=cfg.fpn.upsample_mode,
        )
    else:
        network = SinOut(network, cfg.in_channels, cfg.out_channel)
    return network


class SinOut(nn.Cell):
    def __init__(self, network, in_channels, out_channel):
        super(SinOut, self).__init__()
        self.network = network
        self.out_channel = out_channel
        self.idx = in_channels[-1]
        self.last_conv = nn.Conv2d(in_channels[self.idx], out_channel, kernel_size=1, has_bias=True)

    def construct(self, x):
        x = self.network(x)[self.idx]
        return (self.last_conv(x),)
