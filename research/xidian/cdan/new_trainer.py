from platform import node
from matplotlib.pyplot import axis
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn import LossBase
from platform import node
from matplotlib.pyplot import axis
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn import LossBase
import mindspore.dataset as ds
import copy
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C
from mindspore.ops import functional as F


class CustomWithLossCell_G(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell_G, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label,inputs_source):
        feature, output = self._backbone(data)  # 前向计算得到网络输出
        output_deal = output.narrow(0, 0, inputs_source.shape[0])
        # 强制类型转换
        # int_output_deal = np.argmax(output_deal, axis=1)
        # int_output_deal = Tensor(int_output_deal, dtype=ms.float32)
        label = nn.OneHot(depth=31)(label)
        # return feature, self._loss_fn(output_deal, label)  # 得到标签损失值
        # return feature, self._loss_fn(int_output_deal, label), output
        # print(output.shape)
        # print(label.shape)
        loss = self._loss_fn(output_deal, label)
        # loss = loss.mean(keep_dims=True)
        #  cdan
        # softmax_output = nn.Softmax(axis=1)(output_deal)

        # loss += loss_func.CDAN([feature, softmax_output], self._ad_net, None, None, None)


        return loss

class CustomTrainOneStepCell_G(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer, sens=1.0):
        """入参有三个：训练网络，优化器和反向传播缩放比例"""
        super(CustomTrainOneStepCell_G, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        self.network.set_grad()  # 构建反向网络
        self.optimizer = optimizer  # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        # feature, loss, output = self.network(*inputs)  # 执行前向网络，计算当前输入的损失函数值
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        loss = ops.depend(loss, self.optimizer(grads))  # 使用优化器更新梯度
        # return feature, loss
        return loss
