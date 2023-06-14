import math
import mindspore as ms
from mindspore import ops, nn
from .roi_extractor import RoIExtractor
from mindspore.common.initializer import HeUniform
from ..utils.box_utils import bbox2delta, delta2bbox, shape_prod


class MaskFeat(nn.Cell):
    """
    Feature extraction in Mask head

    Args:
        in_channel (int): Input channels
        out_channel (int): Output channels
        num_convs (int): The number of conv layers, default 4
        norm_type (string | None): Norm type, bn, gn, sync_bn are available,
            default None
    """

    def __init__(self, in_channel=256, out_channel=256, num_convs=4):
        super(MaskFeat, self).__init__()
        self.num_convs = num_convs
        self.in_channel = in_channel
        self.out_channel = out_channel

        mask_conv = nn.SequentialCell()
        for i in range(self.num_convs):
            conv = nn.Conv2d(
                in_channels=in_channel if i == 0 else out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
                pad_mode="pad",
                weight_init=HeUniform(math.sqrt(5)),
                has_bias=True,
            )
            mask_conv.append(conv)
            mask_conv.append(nn.ReLU())

        mask_conv.append(
            nn.Conv2dTranspose(
                in_channels=self.out_channel if num_convs > 0 else self.in_channel,
                out_channels=self.out_channel,
                kernel_size=2,
                stride=2,
                pad_mode="valid",
                weight_init=HeUniform(math.sqrt(5)),
                has_bias=True,
            )
        )
        mask_conv.append(nn.ReLU())
        self.upsample = mask_conv

    def construct(self, feats):
        return self.upsample(feats)


class MaskHead(nn.Cell):
    """RCNN bbox head"""

    def __init__(self, cfg, num_classes):
        super(MaskHead, self).__init__()
        self.num_classes = num_classes
        self.head = MaskFeat(
            in_channel=cfg.head.in_channel, out_channel=cfg.head.out_channel, num_convs=cfg.head.num_convs
        )
        self.roi_extractor = RoIExtractor(cfg.resolution, cfg.roi_extractor.featmap_strides)
        self.mask_fcn_logits = nn.Conv2d(
            in_channels=cfg.head.out_channel,
            out_channels=num_classes,
            kernel_size=1,
            weight_init=HeUniform(math.sqrt(5)),
        )
        self.num_classes = num_classes

    def construct(self, feats, rois, gt_masks, weights, valid_masks):
        """
        feats (list[Tensor]): Feature maps from backbone
        rois (Tensor): select rois, shape is (B, N, 4)
        gt_masks (Tensor): select gt masks, shape is (B, N, R1, R2)
        weights (Tensor): foreground position mask, shape is (B*N, 80, 1, 1)
        valid_masks (Tensor): foreground and background position mask, shape is (B, N,)
        """

        rois_feat = self.roi_extractor(feats, rois, valid_masks)
        _, _, L, R1, R2 = rois_feat.shape
        mask_feat = self.head(rois_feat.reshape(-1, L, R1, R2))
        mask_logits = self.mask_fcn_logits(mask_feat)  # B*N, 80, 28, 28
        mask_logits = ops.sigmoid(mask_logits)
        N, C, H, W = mask_logits.shape
        valid_gt = ops.reduce_max(gt_masks.reshape(N, H, W), (1, 2)) > 0
        weights = ops.select(ops.tile(valid_gt.reshape(-1, 1, 1, 1), (1, C, 1, 1)),
                             weights,
                             ops.zeros_like(weights))
        gt_masks = ops.tile(gt_masks.reshape(N, 1, H, W), (1, C, 1, 1))
        loss_mask = ops.binary_cross_entropy(mask_logits, gt_masks, reduction="none")
        loss_mask = (weights * loss_mask).mean((2, 3))
        loss_mask = loss_mask.sum() / (valid_masks.astype(loss_mask.dtype).sum() + 1e-4)
        return loss_mask

    def predict(self, feats, rois, rois_mask):
        rois_feat = self.roi_extractor(feats, rois, rois_mask)
        B, N, C, R1, R2 = rois_feat.shape
        mask_feat = self.head(rois_feat.reshape(B * N, C, R1, R2))
        mask_logits = self.mask_fcn_logits(mask_feat)  # B*N, 80, 28, 28
        return ops.sigmoid(mask_logits)
