import numpy as np
from mindspore import nn, Tensor
from typing import Type, Union, List, Optional
from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.utils.model_urls import model_urls
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindspore import context
import mindspore.dataset as ds

class ResidualBlock(nn.Cell):
    expansion = 4 #最后一个卷积和的数量是第一个卷积核数量的4倍

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d

        self.conv1 = ConvNormActivation(in_channel, out_channel,
                                        kernel_size=1, norm=norm)
        self.conv2 = ConvNormActivation(out_channel, out_channel,
                                        kernel_size=3, stride=stride, norm=norm)
        self.conv3 = ConvNormActivation(out_channel, out_channel * self.expansion,
                                        kernel_size=1, norm=norm, activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x  # shortscuts分支

        out = self.conv1(x)  # 主分支第一层：1*1卷积层
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.conv3(out)  # 主分支第三层：1*1卷积层

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out

def make_layer(last_out_channel, block: Type[Union[ResidualBlock]],
               channel: int, block_nums: int, stride: int = 1):
    down_sample = None #shortscuts分支

    if stride != 1 or last_out_channel != channel * block.expansion:
        down_sample = ConvNormActivation(last_out_channel, channel * block.expansion,
                                         kernel_size=1, stride=stride, norm=nn.BatchNorm2d, activation=None)
    layers = []
    layers.append(block(last_out_channel, channel, stride=stride, down_sample=down_sample, norm=nn.BatchNorm2d))

    in_channel = channel * block.expansion
    for _ in range(1, block_nums):
        layers.append(block(in_channel, channel, norm=nn.BatchNorm2d))

    return nn.SequentialCell(layers)

class ResNet(nn.Cell):
    def __init__(self,block:Type[Union[ResidualBlock]],
                 layer_nums: List[int], norm:Optional[nn.Cell] = None) -> None:
        super(ResNet, self).__init__()

        if not norm:
            norm = nn.BatchNorm2d

            #第一个卷积层，输入channel为3，输出channel为64
            self.conv1 = ConvNormActivation(3, 64, kernel_size=7, stride=2, norm=norm)

            #最大池化层，缩小图片的尺寸
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

            '''各个残差网络结构块的定义，第一个参数是上一个残差结构的输出channel数，
            block:残差网络的类别，这里都是ResidualBlock
            channel:当前残差结构块的通道数（每个结构的输出通道数一定是channel * 4）
            block_nums:当前残差结构块的堆叠个数
            stride:当前残差结构块的步长
            '''
            self.layer1 = make_layer(64, block, 64, layer_nums[0])
            self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
            self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
            self.layer4 = make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def _resnet(arch:str, block:Type[Union[ResidualBlock]],
            layers: List[int],  pretrained: bool):

    backbone = ResNet(block, layers)
    neck = GlobalAvgPooling() #全局平均池化层
    #将backbone和neck组合成一个网络
    model = BaseClassifier(backbone,neck)

    if pretrained:
        #下载并加载预训练模型
        LoadPretrainedModel(model, model_urls[arch]).run()

        return model
def resnet50(pretrained: bool = False):
    return _resnet('resnet50', ResidualBlock, [3, 4, 6, 3], pretrained)

class DANNet(nn.Cell):
    def __init__(self,num_classes:31):
        super(DANNet, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.fc = nn.Dense(2048,num_classes)

    def construct(self,x):
        x = self.resnet50(x)
        x = self.fc(x)
        return x
#         loss = 0
#         source = self.resnet50(source)
#         if self.training:
#             target = self.resnet50(target)
#             loss += mmd.mmd_rbf_noaccelerate(source, target)

#         source = self.fc(source)

#         return source, loss
    
def create_dataset_imagenet(dataset_path):
    """数据加载"""
    data_set = ds.ImageFolderDataset(dataset_path,
                                     num_parallel_workers=4,
                                     shuffle=True,
                                     decode=True)

    # 数据增强操作
    transform_img = [
        ds.vision.c_transforms.Resize((256,256)),
        ds.vision.c_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
        ds.vision.c_transforms.RandomResizedCrop((224,224)),
        ds.vision.c_transforms.HWC2CHW(),
        # lambda x: ((x / 255).astype("float32"), np.random.normal(size=(100, 1, 1)).astype("float32"))
        ]

    # 数据映射操作
    data_set = data_set.map(input_columns="image",
                            num_parallel_workers=4,
                            operations=transform_img)

    # 批量操作
    data_set = data_set.batch(32, drop_remainder=True)
    return data_set

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    model = resnet50(pretrained=True)
    # print(model)
    dset_loaders = create_dataset_imagenet('../data/office/domain_adaptation_images/amazon/images')
    ds_source = dset_loaders.create_dict_iterator()
    batch_source = next(ds_source)
    image = batch_source['image']
    output = model(image)
    print(output.shape)
#     # model = resnet50(pretrained=True)
#     # print(model)
#     x = Tensor(np.random.rand(32, 3, 224, 224).astype(np.float32))
#     #打印进入DANNet的x的shape
#     y = Tensor(np.random.rand(32, 3, 224, 224).astype(np.float32))

    model = DANNet(31)
    output = model(image)
    print(output.shape)
#     print(source.shape)
#     print(loss)



