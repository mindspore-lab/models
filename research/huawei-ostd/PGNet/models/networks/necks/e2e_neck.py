from mindspore import nn
from models.networks.resnet_cells import ConvNormLayer, DeConvNormLayer


class E2eFpn(nn.Cell):
    """
    PGNet Feature Pyramid Network (FPN) module for text detection.

    Args:
        in_channels (List[int]): The input channel dimensions for feature map.
        out_channels (int): The output channel size.

    Returns:
        Tensor: The output feature map of shape [batch_size, out_channels, H, W].

    """
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        assert len(in_channels) == 7
        self.out_channels = 128

        self.relu = nn.ReLU()
        
        self.conv_bn_layer_1 = ConvNormLayer(3, 32, kernel_size=3, stride=1, padding=1, pad_mode="pad")
        self.conv_bn_layer_2 = ConvNormLayer(64, 64, kernel_size=3, stride=1, padding=1, pad_mode="pad")
        self.conv_bn_layer_3 = ConvNormLayer(256, 128, kernel_size=3, stride=1, padding=1, pad_mode="pad")
        self.conv_bn_layer_4 = ConvNormLayer(32, 64, kernel_size=3, stride=2, padding=1, pad_mode="pad")
        self.conv_bn_layer_5 = ConvNormLayer(64, 64, kernel_size=3, stride=1, padding=1, pad_mode="pad", act=True)
        self.conv_bn_layer_6 = ConvNormLayer(64, 128, kernel_size=3, stride=2, padding=1, pad_mode="pad")
        self.conv_bn_layer_7 = ConvNormLayer(128, 128, kernel_size=3, stride=1, padding=1, pad_mode="pad", act=True)
        self.conv_bn_layer_8 = ConvNormLayer(128, 128, kernel_size=1, stride=1)
        
        num_inputs = [2048, 2048, 1024, 512, 256]
        num_outputs = [256, 256, 192, 192, 128]
        self.conv_h0 = ConvNormLayer(num_inputs[0], num_outputs[0], kernel_size=1, stride=1)
        self.conv_h1 = ConvNormLayer(num_inputs[1], num_outputs[1], kernel_size=1, stride=1)
        self.conv_h2 = ConvNormLayer(num_inputs[2], num_outputs[2], kernel_size=1, stride=1)
        self.conv_h3 = ConvNormLayer(num_inputs[3], num_outputs[3], kernel_size=1, stride=1)
        self.conv_h4 = ConvNormLayer(num_inputs[4], num_outputs[4], kernel_size=1, stride=1)

        self.dconv0 = DeConvNormLayer(num_outputs[0], num_outputs[0 + 1], kernel_size=4, stride=2, padding=1, pad_mode="pad")
        self.dconv1 = DeConvNormLayer(num_outputs[1], num_outputs[1 + 1], kernel_size=4, stride=2, padding=1, pad_mode="pad")
        self.dconv2 = DeConvNormLayer(num_outputs[2], num_outputs[2 + 1], kernel_size=4, stride=2, padding=1, pad_mode="pad")
        self.dconv3 = DeConvNormLayer(num_outputs[3], num_outputs[3 + 1], kernel_size=4, stride=2, padding=1, pad_mode="pad")

        self.conv_g1 = ConvNormLayer(num_outputs[1], num_outputs[1], kernel_size=3, stride=1, padding=1, pad_mode="pad", act=True)
        self.conv_g2 = ConvNormLayer(num_outputs[2], num_outputs[2], kernel_size=3, stride=1, padding=1, pad_mode="pad", act=True)
        self.conv_g3 = ConvNormLayer(num_outputs[3], num_outputs[3], kernel_size=3, stride=1, padding=1, pad_mode="pad", act=True)
        self.conv_g4 = ConvNormLayer(num_outputs[4], num_outputs[4], kernel_size=3, stride=1, padding=1, pad_mode="pad", act=True)

        self.convf = ConvNormLayer(num_outputs[4], num_outputs[4], kernel_size=1, stride=1)

    def construct(self, x):
        c0, c1, c2, c3, c4, c5, c6 = x
        f = [c0, c1, c2]
        h = [None, None, None]
        g = [None, None, None]
        h[0] = self.conv_bn_layer_1(f[0])
        h[1] = self.conv_bn_layer_2(f[1])
        h[2] = self.conv_bn_layer_3(f[2])

        g[0] = self.conv_bn_layer_4(h[0])
        g[1] = self.relu(g[0] + h[1])
        g[1] = self.conv_bn_layer_5(g[1])
        g[1] = self.conv_bn_layer_6(g[1])

        g[2] = self.relu(g[1] + h[2])
        
        g[2] = self.conv_bn_layer_7(g[2])
        f_down = self.conv_bn_layer_8(g[2])

        f1 = [c6, c5, c4, c3, c2]
        h = [None, None, None, None, None]
        g = [None, None, None, None, None]

        h[0] = self.conv_h0(f1[0])
        h[1] = self.conv_h1(f1[1])
        h[2] = self.conv_h2(f1[2])
        h[3] = self.conv_h3(f1[3])
        h[4] = self.conv_h4(f1[4])


        g[0] = self.dconv0(h[0])
        g[1] = self.relu(g[0] + h[1])
        g[1] = self.conv_g1(g[1])
        g[1] = self.dconv1(g[1])

        g[2] = self.relu(g[1] + h[2])
        g[2] = self.conv_g2(g[2])
        g[2] = self.dconv2(g[2])

        g[3] = self.relu(g[2] + h[3])
        g[3] = self.conv_g3(g[3])
        g[3] = self.dconv3(g[3])

        g[4] = self.relu(g[3] + h[4])
        g[4] = self.conv_g4(g[4])
        f_up = self.convf(g[4])
        
        f_common = self.relu(f_down + f_up)
        return f_common
