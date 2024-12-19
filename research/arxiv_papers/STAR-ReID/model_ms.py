import mindspore
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.communication import init
import mindspore.nn as nn
import numpy as np
from resnet_ms import resnet50


class visible_module(nn.Cell):
    def __init__(self, arch="resnet50"):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def construct(self, x):
        x = self.visible.conv1(x)
        x = self.visible.norm(x)
        x = self.visible.relu(x)
        x = self.visible.max_pool(x)
        return x


class thermal_module(nn.Cell):
    def __init__(self, arch="resnet50"):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t

    def construct(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.norm(x)
        x = self.thermal.relu(x)
        x = self.thermal.max_pool(x)
        return x


def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.shape[:2]
    features_mean = (
        features.reshape(batch_size, c, -1).mean(axis=2).reshape(batch_size, c, 1, 1)
    )
    features_std = (
        features.reshape(batch_size, c, -1).std(axis=2).reshape(batch_size, c, 1, 1)
        + 1e-6
    )
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization
    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = (
        style_std * (content_features - content_mean) / content_std + style_mean
    )
    return normalized_features


class gcn_resnet(nn.Cell):
    def __init__(self, arch="resnet50", return_feature_maps=False):
        super(gcn_resnet, self).__init__()

        model_base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.return_feature_maps = return_feature_maps

    def construct(self, x, task="training"):
        x1 = self.base.layer1(x)  # [256, 72, 36]
        x2 = self.base.layer2(x1)  # [512, 36, 18]
        x3 = self.base.layer3(x2)  # [1024, 18, 9]
        x4 = self.base.layer4(x3)  # [2048, 18, 9]

        if self.return_feature_maps:
            return x4, x

        if task == "training":
            dis_x2 = self.same_modal_disturbance(x2)
            dis_x3 = self.base.layer3(dis_x2)
            dis_x3 = self.cross_modal_disturbance(dis_x3)
            x_dis = self.base.layer4(dis_x3)
            return x4, x_dis
        else:
            return x4

    import mindspore.ops as ops

    def same_modal_disturbance(self, x):
        B = x.shape[0]  # 获取 batch 大小
        half_B = B // 2  # 整除操作替代

        x_v = x[:half_B]
        x_t = x[half_B:]

        noise_v = x_v[ops.randperm(x_v.shape[0])]  # 随机选择一个干扰帧
        noise_t = x_t[ops.randperm(x_t.shape[0])]

        distur_v = adain(x_v, noise_v)
        distur_t = adain(x_t, noise_t)
        distur_x = ops.Concat(axis=0)((distur_v, distur_t))  # 合并 disturbed 数据

        return distur_x

    def cross_modal_disturbance(self, x):
        B = x.shape[0]  # 获取 batch 大小
        half_B = B // 2  # 整除操作替代

        x_v = x[:half_B]
        x_t = x[half_B:]

        noise_v = x_v[ops.randperm(x_v.shape[0])]
        noise_t = x_t[ops.randperm(x_t.shape[0])]

        distur_v = adain(x_v, noise_t)
        distur_t = adain(x_t, noise_v)
        distur_x = ops.Concat(axis=0)((distur_v, distur_t))  # 合并 disturbed 数据

        return distur_x


class Normalize(nn.Cell):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def construct(self, x):
        norm = x.pow(self.power).sum(1, keepdims=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class EmbedNet(nn.Cell):
    def __init__(self, class_num, drop=0.2, arch="resnet50", return_feature_maps=False):
        super(EmbedNet, self).__init__()

        # hyper parameters
        pool_dim = 2048
        num_nodes = 17
        seq_length = 12
        self.dropout = drop

        # feature extract
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = gcn_resnet(
            arch=arch, return_feature_maps=return_feature_maps
        )

        # classification layers
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.local_bottleneck = nn.BatchNorm1d(pool_dim, affine=False)
        self.local_classifier = nn.Dense(pool_dim, class_num, has_bias=False)
        self.global_bottleneck = nn.BatchNorm1d(pool_dim, affine=False)
        self.global_classifier = nn.Dense(pool_dim, class_num, has_bias=False)

        self.bn = nn.BatchNorm1d(12, affine=False)
        self.lstm_edge = nn.LSTM(64, 1024, 1, batch_first=True, bidirectional=True)

    def construct(self, x1, x2, p1, p2, modal=0, seq_len=12):
        b, c, h, w = x1.shape
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)

        # style augmentation
        if self.training:
            # IR modality

            frame_batch = seq_len * b
            delta = ops.rand(frame_batch) + 0.5 * ops.ones(frame_batch)  # [0.5-1.5]
            inter_map = delta.unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
            x2 = x2 * inter_map

            # RGB modality
            alpha = (
                (ops.rand(frame_batch) + 0.5 * ops.ones(frame_batch))
                .unsqueeze(dim=1)
                .unsqueeze(dim=1)
                .unsqueeze(dim=1)
            )
            beta = (
                (ops.rand(frame_batch) + 0.5 * ops.ones(frame_batch))
                .unsqueeze(dim=1)
                .unsqueeze(dim=1)
                .unsqueeze(dim=1)
            )
            gamma = (
                (ops.rand(frame_batch) + 0.5 * ops.ones(frame_batch))
                .unsqueeze(dim=1)
                .unsqueeze(dim=1)
                .unsqueeze(dim=1)
            )
            inter_map = ops.cat((alpha, beta, gamma), axis=1)
            x1 = x1 * inter_map
            for i in range(x1.shape[0]):
                x1[i] = x1[i, ops.randperm(3), :, :]

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = ops.cat((x1, x2), 0)

            pose = ops.cat((p1, p2), 0)

        elif modal == 1:
            x = self.visible_module(x1)

        elif modal == 2:
            x = self.thermal_module(x2)

        if self.training:
            x, x_dis = self.base_resnet(x)
        else:
            x = self.base_resnet(x, task="testing")

        if self.training:

            x_local = self.avgpool(x).squeeze()

            b = x_local.shape[0] // seq_len  # 计算 batch 大小
            n = x_local.shape[1]  # 获取特征维度 n
            x_local_2 = x_local.reshape(b, seq_len, n)  # 调整形状

            x_dis_local = self.avgpool(x_dis).squeeze()

            b = x_dis_local.shape[0] // seq_len  # 计算 batch 大小
            n = x_dis_local.shape[1]  # 获取特征维度 n
            x_dis_local_2 = x_dis_local.reshape(b, seq_len, n)  # 调整形状

            x_local_feat = self.local_bottleneck(x_local)
            x_local_logits = self.local_classifier(x_local_feat)

            x_dis_feat = self.local_bottleneck(x_dis_local)
            x_dis_logits = self.local_classifier(x_dis_feat)

            p = 3.0
            x_global = (ops.mean(x_local_2**p, axis=1) + 1e-12) ** (1 / p)

            global_feat = self.global_bottleneck(x_global)
            logits = self.global_classifier(global_feat)

            defense_loss = ops.mean(ops.sqrt((x_local - x_dis_local).pow(2).sum(1)))

            return x_global, x_local, logits, x_local_logits, x_dis_logits, defense_loss
        else:

            x_local = self.avgpool(x).squeeze()
            b = x_local.shape[0] // seq_len  # 计算 batch 大小
            n = x_local.shape[1]  # 获取特征维度 n
            x_local_2 = x_local.reshape(b, seq_len, n)  # 调整形状

            p = 3.0
            x_global = (ops.mean(x_local_2**p, axis=1) + 1e-12) ** (1 / p)

            global_feat = self.global_bottleneck(x_global)
            return self.l2norm(global_feat)


if __name__ == "__main__":
    # 设置运行模式和设备
    # mindspore.set_context(mode=context.PYNATIVE_MODE)
    mindspore.set_auto_parallel_context(
        parallel_mode=mindspore.ParallelMode.AUTO_PARALLEL, gradients_mean=True
    )
    init("hccl")
    # mindspore.set_seed(1)
    # 定义输入数据
    input1 = Tensor(np.random.randn(2, 36, 288, 144), mindspore.float32)
    input2 = Tensor(np.random.randn(2, 36, 288, 144), mindspore.float32)
    input3 = Tensor(np.random.randn(2, 12, 17, 3), mindspore.float32)
    input4 = Tensor(np.random.randn(2, 12, 17, 3), mindspore.float32)

    # 初始化网络并进行训练模式下的前向测试
    net = EmbedNet(class_num=500, drop=0.2, arch="resnet50")
    net.set_train(True)
    x_global, x_local, logits, x_local_logits, x_dis_logits, defense_loss = net(
        input1, input2, input3, input4, modal=0, seq_len=12
    )

    print("-----------------------------------")
    print(x_global.shape)
    print(x_local.shape)
    print(logits.shape)
    print(x_local_logits.shape)
    print(x_dis_logits.shape)
    print(defense_loss)
    print("Model train has been tested successfully!")

    print("-----------------------------------")

    # 切换到评估模式并进行前向测试
    net.set_train(False)
    global_feat1 = net(input1, input2, input3, input4, modal=1, seq_len=12)
    global_feat2 = net(input1, input2, input3, input4, modal=2, seq_len=12)

    print(global_feat1.shape)
    print(global_feat2.shape)
    print("Model eval has been tested successfully!")
    print("-----------------------------------")
