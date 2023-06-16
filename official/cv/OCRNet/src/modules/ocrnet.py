import math
from mindspore.common.initializer import HeUniform
from mindspore import ops, nn
from .backbone import create_backbone
from .base_modules import ConvModule, FCNHead


class SpatialGatherModule(nn.Cell):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def construct(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.shape
        channels = feats.shape[1]
        probs = probs.reshape((batch_size, num_classes, -1))
        feats = feats.reshape((batch_size, channels, -1))
        # [batch_size, height*width, num_classes]
        feats = feats.transpose((0, 2, 1))
        # [batch_size, channels, height*width]
        probs = ops.softmax(self.scale * probs, axis=2)
        # [batch_size, channels, num_classes]
        ocr_context = ops.matmul(probs, feats)
        ocr_context = ocr_context.transpose((0, 2, 1)).unsqueeze(3)
        return ocr_context


class SelfAttentionBlock(nn.Cell):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(
        self,
        key_in_channels,
        query_in_channels,
        channels,
        out_channels,
        share_key_query,
        query_downsample,
        key_downsample,
        key_query_num_convs,
        value_out_num_convs,
        key_query_norm,
        value_out_norm,
        matmul_norm,
        with_out,
        norm,
        act,
    ):
        super().__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.norm = norm
        self.act = act
        self.key_project = self.build_project(
            key_in_channels, channels, num_convs=key_query_num_convs, use_conv_module=key_query_norm, norm=norm, act=act
        )
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                norm=norm,
                act=act,
            )
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            norm=norm,
            act=act,
        )
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                norm=norm,
                act=act,
            )
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

    def build_project(self, in_channels, channels, num_convs, use_conv_module, norm, act):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [ConvModule(in_channels, channels, 1, norm=norm, act=act)]
            for _ in range(num_convs - 1):
                convs.append(ConvModule(channels, channels, 1, norm=norm, act=act))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1, weight_init=HeUniform(math.sqrt(5)))]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1, weight_init=HeUniform(math.sqrt(5))))
        if len(convs) > 1:
            convs = nn.SequentialCell(convs)
        else:
            convs = convs[0]
        return convs

    def construct(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.shape[0]
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(query.shape[0], query.shape[1], -1)
        query = query.transpose((0, 2, 1))

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(key.shape[0], key.shape[1], -1)
        value = value.reshape(value.shape[0], value.shape[1], -1)
        value = value.transpose((0, 2, 1))

        sim_map = ops.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-0.5) * sim_map
        sim_map = ops.softmax(sim_map, axis=-1)

        context = ops.matmul(sim_map, value)
        context = context.transpose((0, 2, 1))
        context = context.reshape((batch_size, -1, query_feats.shape[2], query_feats.shape[3]))
        if self.out_project is not None:
            context = self.out_project(context)
        return context


class ObjectAttentionBlock(nn.Cell):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, norm, act):
        super(ObjectAttentionBlock, self).__init__()
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        self.self_att = SelfAttentionBlock(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            norm=norm,
            act=act,
        )
        self.bottleneck = ConvModule(in_channels * 2, channels, 3, padding=1, norm=norm, act=act)

    def construct(self, query_feats, key_feats):
        """Forward function."""
        context = self.self_att(query_feats, key_feats)
        output = self.bottleneck(ops.concat((context, query_feats), axis=1))
        return output


class OCRHead(nn.Cell):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(
        self,
        channels,
        in_channels,
        ocr_channels,
        out_channels=None,
        in_index=(0, 1, 2, 3),
        num_classes=2,
        scale=1,
        align_corners=None,
        norm="none",
        act="relu",
    ):
        super(OCRHead, self).__init__()
        if isinstance(in_channels, (list, tuple)):
            self.in_channels = sum(in_channels)
        else:
            self.in_channels = in_channels
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(channels, self.ocr_channels, self.scale, norm=norm, act=act)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(self.in_channels, channels, 3, padding=1, norm=norm, act=act)
        if out_channels is None:
            self.out_channels = num_classes
        self.conv_seg = nn.Conv2d(
            ocr_channels, self.out_channels, kernel_size=1, weight_init=HeUniform(math.sqrt(5)), has_bias=True
        )
        self.in_index = in_index
        self.align_corners = align_corners

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        upsampled_inputs = ()
        for idx in self.in_index:
            inp = inputs[idx]
            inp = ops.interpolate(inp, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            upsampled_inputs += (inp,)

        inputs = ops.concat(upsampled_inputs, axis=1)
        return inputs

    def construct(self, inputs, prev_output):
        """Forward function."""
        x = self._forward_feature(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = self.conv_seg(object_context)

        return output


class OCRNet(nn.Cell):
    def __init__(self, cfg):
        super(OCRNet, self).__init__()
        self.backbone = create_backbone(
            initializer=cfg.backbone.initializer,
            in_channels=cfg.backbone.in_channels,
            pretrained=cfg.backbone.pretrained,
            backbone_ckpt=cfg.backbone.backbone_ckpt,
        )
        self.fcn_head = FCNHead(
            in_channels=cfg.fcn_head.in_channels,
            channels=cfg.fcn_head.channels,
            num_classes=cfg.num_classes,
            kernel_size=cfg.fcn_head.kernel_size,
            num_convs=cfg.fcn_head.num_convs,
            concat_input=cfg.fcn_head.concat_input,
            in_index=cfg.fcn_head.in_index,
            norm=cfg.fcn_head.norm,
            act=cfg.fcn_head.act,
        )
        self.ocr_head = OCRHead(
            in_channels=cfg.ocr_head.in_channels,
            channels=cfg.ocr_head.channels,
            num_classes=cfg.num_classes,
            in_index=cfg.ocr_head.in_index,
            ocr_channels=cfg.ocr_head.ocr_channels,
            norm=cfg.ocr_head.norm,
            act=cfg.ocr_head.act,
        )

    def construct(self, img):
        feats = self.backbone(img)
        fcn_out = self.fcn_head(feats)
        ocr_out = self.ocr_head(feats, fcn_out)
        return ocr_out, fcn_out
