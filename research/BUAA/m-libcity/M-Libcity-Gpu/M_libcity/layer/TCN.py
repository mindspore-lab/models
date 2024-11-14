import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np

class Align(nn.Cell):
    """
    # align channel_in and channel_out
    """

    def __init__(self, channel_in, channel_out):
        super(Align, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.align_conv = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_out, kernel_size=1,
                                    pad_mode='valid', weight_init='he_uniform')

    def construct(self, x):
        x_align = x
        if self.channel_in > self.channel_out:
            x_align = self.align_conv(x)
        elif self.channel_in < self.channel_out:
            dim1, dim2, dim3, dim4 = x.shape
            y = ops.Zeros()((dim1, self.channel_out - self.channel_in, dim3, dim4), x.dtype)
            x_align = ops.Concat(axis=1)((x, y))
        return x_align


class TCN(nn.Cell):
    """
    # 1.align
    # 2.conv2d
    # 3.glu or gtu
    """

    def __init__(self, t_kernel_size, channel_in, channel_out, vertex_num, gate_type):
        super(TCN, self).__init__()
        self.t_kernel_size = t_kernel_size
        self.gate_type = gate_type
        self.align = Align(channel_in, channel_out)
        self.conv2d = nn.Conv2d(channel_in, 2 * channel_out, (t_kernel_size, 1), stride=(1, 1),
                                padding=0, pad_mode='valid', dilation=(1, 1), group=1, has_bias=True,
                                weight_init='he_uniform')

    def construct(self, x):
        x_in = self.align(x)
        x_in = x_in[:, :, self.t_kernel_size - 1:, :]
        x_conv2d = self.conv2d(x)
        x_pq = ops.Split(axis=1, output_num=2)(x_conv2d)
        if self.gate_type == 'glu':  # glu
            # (x_p + x_in) ⊙ Sigmoid(x_q)
            x_glu = ops.Mul()(ops.Add()(x_pq[0], x_in), nn.Sigmoid()(x_pq[1]))
            x_tc_out = x_glu
        else:  # self.gate_type == 'gtu'
            # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
            x_gtu = ops.Mul()(nn.Tanh()(ops.Add()(x_pq[0], x_in)), nn.Sigmoid()(x_pq[1]))
            x_tc_out = x_gtu
        return x_tc_out