import numpy as np

import mindspore as ms
from mindspore import Tensor, nn

from model.mobilenet_v1 import MobileNet
from model.yolov3head import YOLOv3Head
from model.yolo_fpn import YOLOv3FPN

__all__ = ["YOLOv3_MobileNet_V1", "yolov3_mobilenet_v1"]


def _cfg(url="", **kwargs):
    return {"url": url, **kwargs}


default_cfgs = {"yolov3": _cfg(url="")}


class YOLOv3_MobileNet_V1(nn.Cell):
    def __init__(self, stride=[32, 16, 8], in_channels=3, num_classes=None):
        super(YOLOv3_MobileNet_V1, self).__init__()
        self.stride = Tensor(np.array(stride), ms.int32)
        self.stride_max = int(max(stride))
        ch, nc = in_channels, num_classes

        self.nc = nc  # override yaml value

        self.backbone = MobileNet(scale=1,
                                feature_maps=[4, 6, 13],
                                with_extra_blocks=False,
                                extra_block_filters=[])
        self.neck = YOLOv3FPN()
        self.yolo_head = YOLOv3Head(anchors=[[10,13, 16,30, 33,23],
                                            [30,61, 62,45, 59,119],
                                            [116,90, 156,198, 373,326]],
                                     stride=stride,
                                     ch=[1024, 512, 256])

    def construct(self, x):
        body_feats = self.backbone(x)
        neck_feats = self.neck(body_feats)
        yolo_head_outs = self.yolo_head(neck_feats)
        return yolo_head_outs


def yolov3_mobilenet_v1(stride=[32, 16, 8], in_channels=3, num_classes=None, **kwargs) -> YOLOv3_MobileNet_V1:
    """Get yolov3 model."""
    model = YOLOv3_MobileNet_V1(stride=stride, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return model
