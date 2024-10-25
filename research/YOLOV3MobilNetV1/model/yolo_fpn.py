from mindspore import nn, ops, jit


def batch_norm(ch,
               norm_type='bn',
               norm_decay=0.,
               freeze_norm=False,
               initializer=None,
               data_format='NCHW'):
    if norm_type in ['sync_bn', 'bn']:
        norm_layer = nn.BatchNorm2d(ch)

    norm_params = norm_layer.trainable_params()
    if freeze_norm:
        for param in norm_params:
            param.requires_grad = False

    return norm_layer


class ConvBNLayer(nn.Cell):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 norm_type='bn',
                 norm_decay=0.,
                 act="leaky",
                 freeze_norm=False,
                 data_format='NCHW',
                 name=''):
        """
        conv + bn + activation layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            stride (int): stride, default 1
            groups (int): number of groups of conv layer, default 1
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            act (str): activation function type, default 'leaky', which means leaky_relu
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            group=groups,
            has_bias=False)
        self.batch_norm = batch_norm(
            ch_out,
            norm_type=norm_type,
            norm_decay=norm_decay,
            freeze_norm=freeze_norm,
            data_format=data_format)
        self.act = act

    def construct(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = ops.leaky_relu(out, 0.1)
        return out


class YoloDetBlock(nn.Cell):
    def __init__(self,
                 ch_in,
                 channel,
                 norm_type,
                 freeze_norm=False,
                 name='',
                 data_format='NCHW'):
        """
        YOLODetBlock layer for yolov3, see https://arxiv.org/abs/1804.02767

        Args:
            ch_in (int): input channel
            channel (int): base channel
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(YoloDetBlock, self).__init__()
        self.ch_in = ch_in
        self.channel = channel
        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
        conv_def = [
            ['conv0', ch_in, channel, 1, '.0.0'],
            ['conv1', channel, channel * 2, 3, '.0.1'],
            ['conv2', channel * 2, channel, 1, '.1.0'],
            ['conv3', channel, channel * 2, 3, '.1.1'],
            ['route', channel * 2, channel, 1, '.2'],
        ]

        self.conv_module_list = []
        self.conv_module = nn.SequentialCell()
        for idx, (conv_name, ch_in, ch_out, filter_size,
                  post_name) in enumerate(conv_def):
            self.conv_module.insert_child_to_cell(
                conv_name,
                ConvBNLayer(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    filter_size=filter_size,
                    padding=(filter_size - 1) // 2,
                    norm_type=norm_type,
                    freeze_norm=freeze_norm,
                    data_format=data_format,
                    name=name + post_name))
            self.conv_module_list.append(getattr(self.conv_module, conv_name))

        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            padding=1,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            data_format=data_format,
            name=name + '.tip')

    def construct(self, inputs):
        route = self.conv_module_list[0](inputs)
        for i in range(1, len(self.conv_module_list)):
            route = self.conv_module_list[i](route)
        tip = self.tip(route)
        return route, tip


class YOLOv3FPN(nn.Cell):
    __shared__ = ['norm_type', 'data_format']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 norm_type='bn',
                 freeze_norm=False,
                 data_format='NCHW'):
        """
        YOLOv3FPN layer

        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC

        """
        super(YOLOv3FPN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)

        self._out_channels = []
        yolo_transition = []
        self.data_format = data_format
        yolo_block = []
        for i in range(self.num_blocks):
            name = 'yolo_block.{}'.format(i)
            in_channel = in_channels[-i - 1]
            if i > 0:
                in_channel += 512 // (2**i)
            yolo_block.append(YoloDetBlock(
                            in_channel,
                            channel=512 // (2**i),
                            norm_type=norm_type,
                            freeze_norm=freeze_norm,
                            data_format=data_format,
                            name=name))
            # tip layer output channel doubled
            self._out_channels.append(1024 // (2**i))

            if i < self.num_blocks - 1:
                name = 'yolo_transition.{}'.format(i)
                yolo_transition.append(
                    ConvBNLayer(
                        ch_in=512 // (2**i),
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        norm_type=norm_type,
                        freeze_norm=freeze_norm,
                        data_format=data_format,
                        name=name))
        self.yolo_block = nn.CellList(yolo_block)
        self.yolo_transition = nn.CellList(yolo_transition)

    def construct(self, blocks):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        yolo_feats = []

        route = ops.zeros_like(blocks[0])
        # add embedding features output for multi-object tracking model
        for i, block in enumerate(blocks):
            if i > 0:
                block = ops.concat([route, block], axis=1)
            route, tip = self.yolo_block[i](block)
            yolo_feats.append(tip)

            if i < self.num_blocks - 1:
                route = self.yolo_transition[i](route)
                route = ops.interpolate(route, (route.shape[-2] * 2, route.shape[-1] * 2), mode='bilinear')

        return yolo_feats

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }
