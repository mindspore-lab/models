import mindspore as ms
from mindspore import ops
from mindcv import create_model
from .hrnet import *


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
            print("Not support get_local_rank, get local_rank by get_rank() % 8")

        if local_rank == 0:
            print(f"==== create_model {local_rank} start")
            net = create_model(
                initializer, in_channels=in_channels, pretrained=pretrained, checkpoint_path=backbone_ckpt
            )
            print(f"==== create_model {local_rank} done")
            r = allreduce_sum(ops.ones((1), ms.float32))
        else:
            r = allreduce_sum(ops.ones((1), ms.float32))
            print(f"==== create_model {local_rank} {r} start")
            net = create_model(
                initializer, in_channels=in_channels, pretrained=pretrained, checkpoint_path=backbone_ckpt
            )
            print(f"==== create_model {local_rank} done")
    else:
        net = create_model(initializer, in_channels=in_channels, pretrained=pretrained, checkpoint_path=backbone_ckpt)
    return net
