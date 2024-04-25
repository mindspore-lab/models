import mindspore
import mindspore.ops as ops
import math
class Dense(mindspore.nn.Cell):
    """patched Dense"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 has_bias=True,
                 dtype=mindspore.common.dtype.float32):
        """Initialize Dense."""
        super().__init__()
        self.in_channels = mindspore._checkparam.check_positive_int(
            in_channels, "in_channels", self.cls_name)
        self.out_channels = mindspore._checkparam.check_positive_int(
            out_channels, "out_channels", self.cls_name)
        self.has_bias = mindspore._checkparam.check_bool(
            has_bias, "has_bias", self.cls_name)

        self.weight = mindspore.Parameter(mindspore.common.initializer.initializer(
            mindspore.common.initializer.HeUniform(math.sqrt(5)), [out_channels, in_channels], dtype=dtype), name="weight")

        self.bias = None
        if self.has_bias:
            fan_in, _ = mindspore.common.initializer._calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias = mindspore.Parameter(mindspore.common.initializer.initializer(
                mindspore.common.initializer.Uniform(bound), [out_channels], dtype=dtype), name="bias")

    def construct(self, x):
        # if LESS_MS_2_2:
        x_shape = x.shape
        if len(x_shape) != 2:
            x = x.reshape(-1, x.shape[-1])
        x = ops.matmul(x, self.weight.T)
        if self.has_bias:
            x = ops.add(x, self.bias)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (x.shape[-1],)
            x = x.reshape(out_shape)
        return x

    def extend_repr(self):
        s = f'input_channels={self.in_channels}, output_channels={self.out_channels}'
        if self.has_bias:
            s += f', has_bias={self.has_bias}'
        return s
mindspore.nn.Dense = Dense