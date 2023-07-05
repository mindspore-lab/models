import math
import mindspore as ms
from mindspore import ops, nn
from .roi_extractor import RoIExtractor
from mindspore.common.initializer import HeUniform


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
                bias_init="zeros"
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
                bias_init="zeros"
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
            has_bias=True,
            bias_init="zeros"
        )
        self.num_classes = num_classes
        self.crop_and_resize = ops.CropAndResize(method="bilinear_v2")
        self.mask_shape = tuple(cfg.mask_shape)

    def construct(self, feats, rois, gt_masks, weights, valid_masks):
        """
        feats (list[Tensor]): Feature maps from backbone
        rois (Tensor): select rois, shape is (B, N, 4), N is pos num
        gt_masks (Tensor): select gt masks, shape is (B, N, H, W), H, W is origin image's height and width.
        weights (Tensor): foreground position mask, shape is (B*N, 80, 1, 1)
        valid_masks (Tensor): foreground and background position mask, shape is (B, N,)
        """
        fg_masks = valid_masks[:, :rois.shape[1]]
        rois_feat = self.roi_extractor(feats, rois, fg_masks)
        B, N, L, R1, R2 = rois_feat.shape
        mask_feat = self.head(rois_feat.reshape(B * N, L, R1, R2))
        mask_logits = self.mask_fcn_logits(mask_feat)  # B*N, 80, 28, 28
        mask_logits = mask_logits.astype(ms.float32)
        mask_logits = ops.sigmoid(mask_logits)
        C = mask_logits.shape[1]
        _, _, H, W = gt_masks.shape
        rois = rois.reshape((B * N ,4))
        x1, y1, x2, y2 = ops.split(rois, split_size_or_sections=1, axis=-1)

        boxes = ops.concat((y1 / H, x1 / W, y2 / H, x2 / W), -1)
        box_ids = ops.arange(B * N).astype(ms.int32)
        gt_masks = self.crop_and_resize(gt_masks.reshape(B * N, H, W, 1).astype(ms.float32),
                                        boxes,
                                        box_ids,
                                        self.mask_shape)
        gt_masks = ops.round(gt_masks)
        gt_masks = ops.tile(gt_masks.reshape(B * N, 1, self.mask_shape[0], self.mask_shape[1]), (1, C, 1, 1))
        gt_masks = ops.stop_gradient(gt_masks)
        loss_mask = ops.binary_cross_entropy(mask_logits, gt_masks, reduction="none")
        loss_mask = weights * loss_mask.mean((2, 3))
        loss_mask = loss_mask.sum() / (fg_masks.astype(loss_mask.dtype).sum() + 1e-4)
        return loss_mask

    def predict(self, feats, rois, rois_mask):
        rois_feat = self.roi_extractor(feats, rois, rois_mask)
        B, N, L, R1, R2 = rois_feat.shape
        mask_feat = self.head(rois_feat.reshape(B * N, L, R1, R2))
        mask_logits = self.mask_fcn_logits(mask_feat)  # B*N, 80, 28, 28
        mask_logits = mask_logits.astype(ms.float32)
        return ops.sigmoid(mask_logits)
