from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from typing import Type, Union, List, Optional
from mindvision.classification.models.blocks import ConvNormActivation
from mindspore import nn
import mindspore.ops as ops
import numpy as np
import mindspore
import mindspore as ms
from mindspore import Tensor
import loss as loss_func
from mindspore.common import initializer
from resnet_backbone.resnet_model import resnet50

#  resnet

class ResNetFc(nn.Cell):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        # model_resnet = models.resnet50(pretrained=True)
        model_resnet = resnet50(pretrained=True)
        self.back_bone = model_resnet.backbone
        self.neck = model_resnet.neck
        # self.head = model_resnet.head
        self.bottleneck = init_weight(nn.Dense(2048, bottleneck_dim))
        self.fc = init_weight(nn.Dense(bottleneck_dim, class_num))
        # self.fc = model_resnet.fc
        self.__in_features = bottleneck_dim
    def construct(self, x):
        x = self.back_bone(x)
        x = self.neck(x)
        x = self.bottleneck(x)  # x为传入GCN的特征(batch,256)
        # print("after neck shape is {}".format(x.shape))
        # x = self.head(x)
        y = self.fc(x)  # y为mlp的logits结果(batch ,num class)

        return x, y

    def output_num(self):
        return self.__in_features


# For SVHN dataset
class DTN(nn.Cell):
    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, pad_mode='pad', padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, pad_mode='pad', padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, pad_mode='pad', padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.fc_params = nn.SequentialCell(
            nn.Dense(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Dense(512, 10)


        self.__in_features = 512

    def construct(self, x):
        x = self.conv_params(x)
        # x = x.view(x.size(0), -1)
        x = nn.Flatten()(x)
        x = self.fc_params(x)
        logits = self.classifier(x)
        # index = logits.argmax(axis=1)
        # print(index)
        # index = index.astype("float32")
        return x, logits

    def output_num(self):
        return self.__in_features




# 定义初始化函数
def init_weight(net):
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer.initializer(initializer.HeUniform(),
                                                     cell.weight.shape,
                                                     cell.weight.dtype))
            cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                       cell.bias.shape,
                                                       cell.bias.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
            cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                       cell.bias.shape,
                                                       cell.bias.dtype))
        if isinstance(cell, nn.BatchNorm2d):
            cell.weight.set_data(initializer.initializer(initializer.Normal(sigma=0.02, mean=1.0),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))
            cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                       cell.bias.shape,
                                                       cell.bias.dtype))
    return net


class RandomLayer(nn.Cell):
    def __init__(self, input_dim_list, output_dim): #input dim list为[256,31]):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim    #1024
        # self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)]
        #self.random_matrix = [ops.StandardNormal()((input_dim_list[i], output_dim)) for i in range(self.input_num)]
        stdnormal = ops.StandardNormal()
        self.random_matrix = [stdnormal((input_dim_list[i], output_dim)) for i in range(self.input_num)]
    def construct(self, input_list):
        return_list = [ops.MatMul()(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        pow = ops.Pow()
        x = Tensor(self.output_dim, mindspore.float16)
        y = Tensor(1. / len(return_list), mindspore.float16)
        output = pow(x, y)
        # print(output)
        return_tensor = return_list[0] / output 
        #return_tensor = return_list[0] / math.pow(float(self.output_dim), 1. / len(return_list))
        for single in return_list[1:]:
            #return_tensor = ops.Mul()(return_tensor, single)
            return_tensor = ops.mul(return_tensor, single)
        return return_tensor


class AdversarialNetwork(nn.Cell):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = init_weight(nn.Dense(in_feature, hidden_size))
        self.ad_layer2 = init_weight(nn.Dense(hidden_size, hidden_size))
        self.ad_layer3 = init_weight(nn.Dense(hidden_size, 1))
        # self.ad_layer1 = nn.Dense(in_feature, hidden_size)
        # self.ad_layer2 = nn.Dense(hidden_size, hidden_size)
        # self.ad_layer3 = nn.Dense(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


    def construct(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    # def get_parameters(self):
    #     return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

    
    
# 连接网络
class CustomWithLossCell_G(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell_G, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, inputs_source):
        feature, output = self._backbone(data)  # 前向计算得到网络输出
        output_deal = output.narrow(0, 0, inputs_source.shape[0])
        # 强制类型转换
        # int_output_deal = np.argmax(output_deal, axis=1)
        # int_output_deal = Tensor(int_output_deal, dtype=ms.float32)
        label = nn.OneHot(depth=10)(label)
        loss = self._loss_fn(output_deal, label)
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
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        loss = ops.depend(loss, self.optimizer(grads))  # 使用优化器更新梯度
        return loss


class CustomWithLossCell_D1(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn, ad_net, random_layer):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell_D1, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._backbone.set_grad(False)
        self._loss_fn = loss_fn
        self._ad_net = ad_net
        self._ad_net.set_grad(True)
        self._random_layer = random_layer

    def construct(self, data, label):
        feature, output = self._backbone(data)  # 前向计算得到网络输出
        softmax_output = nn.Softmax(axis=1)(output)
        if self._random_layer is None:
            tmp1 = ops.expand_dims(softmax_output, 2)
            tmp2 = ops.expand_dims(feature, 1)
            batmatmul = ops.BatchMatMul()
            op_out = batmatmul(tmp1, tmp2)
            # op_out = mindspore.ops.BatchMatMul(x=tmp1, y=tmp2)
            ad_out = self._ad_net(op_out.view(-1, softmax_output.shape[1] * feature.shape[1]))
        else:
            random_out = self._random_layer.construct([feature, softmax_output])
            ad_out =  self._ad_net(random_out.view(-1, random_out.shape[1]))    
        # loss += loss_func.CDAN([feature, softmax_output], self._ad_net, None, None, None)
        batch_size = softmax_output.shape[0] // 2
        tmp = np.array([[1]] * batch_size + [[0]] * batch_size)
        dc_target = mindspore.Tensor.from_numpy(tmp)
        dc_target = mindspore.Tensor(dc_target, dtype=ms.float32)
        # print(ad_out.shape)
        # print(dc_target.shape)
        if ad_out.shape != dc_target.shape:
            print(1)
        
        adv_loss1 = self._loss_fn(ad_out, dc_target)

        return adv_loss1

class CustomTrainOneStepCell_D1(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer, sens=1.0):
        """入参有三个：训练网络，优化器和反向传播缩放比例"""
        super(CustomTrainOneStepCell_D1, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        # self.network.set_grad()  # 构建反向网络
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

class CustomWithLossCell_D2(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn, ad_net, random_layer):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell_D2, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._backbone.set_grad(True)
        self._loss_fn = loss_fn
        self._ad_net = ad_net
        self._ad_net.set_grad(False)
        self._random_layer = random_layer


    def construct(self, data, label):
        feature, output = self._backbone(data)  # 前向计算得到网络输出
        softmax_output = nn.Softmax(axis=1)(output)
        if self._random_layer is None:
            softmax_output = ops.stop_gradient(softmax_output)
            tmp1 = ops.expand_dims(softmax_output, 2)
            tmp2 = ops.expand_dims(feature, 1)
            batmatmul = ops.BatchMatMul()
            op_out = batmatmul(tmp1, tmp2)
            # op_out = mindspore.ops.BatchMatMul(x=tmp1, y=tmp2)
            ad_out = self._ad_net(op_out.view(-1, softmax_output.shape[1] * feature.shape[1]))
            # loss += loss_func.CDAN([feature, softmax_output], self._ad_net, None, None, None)
        else:
            random_out = self._random_layer.construct([feature, softmax_output])
            ad_out =  self._ad_net(random_out.view(-1, random_out.shape[1]))
            
        batch_size = softmax_output.shape[0] // 2
        tmp = np.array([[0]] * batch_size + [[1]] * batch_size)
        dc_target = mindspore.Tensor.from_numpy(tmp)
        dc_target = mindspore.Tensor(dc_target, dtype=ms.float32)
        adv_loss2 = self._loss_fn(ad_out, dc_target)

        return adv_loss2

class CustomTrainOneStepCell_D2(nn.Cell):
    """自定义训练网络"""

    def __init__(self, network, optimizer, sens=1.0):
        """入参有三个：训练网络，优化器和反向传播缩放比例"""
        super(CustomTrainOneStepCell_D2, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        # self.network.set_grad()  # 构建反向网络
        self.optimizer = optimizer  # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True)  # 反向传播获取梯度

    def construct(self, *inputs):
        # 执行前向网络，计算当前输入的损失函数值
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)  # 进行反向传播，计算梯度
        loss = ops.depend(loss, self.optimizer(grads))  # 使用优化器更新梯度
        return loss

