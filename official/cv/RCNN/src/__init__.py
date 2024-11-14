from .faster_rcnn import FasterRCNN
from .mask_rcnn import MaskRCNN


def get_network(cfg):
    if cfg.net == "FasterRCNN":
        return FasterRCNN(cfg)
    if cfg.net == "MaskRCNN":
        return MaskRCNN(cfg)
    raise InterruptedError
