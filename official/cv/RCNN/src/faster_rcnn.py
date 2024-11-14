import mindspore as ms
from mindspore import ops, nn

from src.backbone import build_backbone
from src.rpn_head import RPNHead
from src.bbox_head import BBoxHead


class FasterRCNN(nn.Cell):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        self.backbone = build_backbone(cfg.backbone)
        backbone_feat_nums = 1
        if hasattr(cfg.backbone, "fpn"):
            backbone_feat_nums = cfg.backbone.fpn.num_outs
        self.rpn_head = RPNHead(
            cfg.rpn_head, backbone_feat_nums=backbone_feat_nums, in_channel=cfg.backbone.out_channels
        )
        self.bbox_head = BBoxHead(cfg.bbox_head, num_classes=cfg.data.nc)

    def construct(self, images, gt_bbox, gt_class):
        gts = ops.concat((gt_class.astype(gt_bbox.dtype), gt_bbox), -1)
        features = self.backbone(images)
        image_shape = images.shape[2:]
        rois, rois_mask, loss_rpn_cls, loss_rpn_reg = self.rpn_head(features, gts, image_shape)
        loss_bbox_reg, loss_bbox_cls = self.bbox_head(features, rois, rois_mask, gts)
        loss = loss_rpn_cls + loss_rpn_reg + loss_bbox_cls + loss_bbox_reg
        return loss, loss_rpn_cls + loss_rpn_reg, loss_bbox_cls + loss_bbox_reg

    @ms.jit
    def predict(self, images):
        features = self.backbone(images)
        image_shape = images.shape[2:]
        rois, rois_mask = self.rpn_head.predict(features, image_shape)
        res = self.bbox_head.predict(features, rois, rois_mask)
        return res
