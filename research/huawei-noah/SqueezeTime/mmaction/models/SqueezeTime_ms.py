import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore import context
import os
from mmaction.registry import MODELS
def get_inplanes():
    return [64, 128, 256, 512]

def normal_init(m,mean=0,std=1,bias=0):
    m.weight.set_data(init.initializer(init.Normal(std, mean), m.weight.shape))
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.set_data(init.initializer(init.Constant(bias), m.bias.shape))
class SpatialConv(nn.Cell):
    def __init__(self,dim_in,dim_out,pos_dim=7):
        super(SpatialConv,self).__init__()
        self.short_conv=nn.Conv2d(dim_in,dim_out,kernel_size=3,stride=1,padding=1,has_bias=True,group=1,pad_mode='pad')
        self.glo_conv=nn.SequentialCell(nn.Conv2d(dim_in,16,kernel_size=3,stride=1,padding=1,group=1,pad_mode='pad',has_bias=True),nn.BatchNorm2d(16),nn.ReLU(),nn.Conv2d(16,16,kernel_size=7,stride=1,padding=3,pad_mode='pad',has_bias=True),nn.BatchNorm2d(16),nn.ReLU(),nn.Conv2d(16,dim_out,kernel_size=3,stride=1,padding=1,group=1,pad_mode='pad',has_bias=True),nn.Sigmoid())
        self.pos_embed=ms.Parameter(ops.Zeros()((1,16,pos_dim,pos_dim),ms.float32))
        self.pos_embed=self.pos_embed.set_data(init.initializer(init.HeNormal(),(1,16,pos_dim,pos_dim),ms.float32))
    def construct(self,x,param):

        x_short=self.short_conv(x)
        x=x*param
        for i in range(len(self.glo_conv)):
            if i==3:
                _,_,H,W=x.shape
                if self.pos_embed.shape[2] !=H or self.pos_embed.shape[3]!=W:
                    pos_embed=ops.interpolate(self.pos_embed,scales=None, sizes=(H,W), coordinate_transformation_mode='align_corners', mode='bilinear')
                else:
                    pos_embed=self.pos_embed
                x=x+pos_embed
            x=self.glo_conv[i](x)
        
        return x_short*x
class Conv2d(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups= 1,
        bias=True,
        padding_mode = 'pad', 
        ghost_ratio=0.5,pos_dim=7):
        super(Conv2d,self).__init__()
        self.stride=stride
        self.param_conv=nn.SequentialCell(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels,
                      in_channels,
                      1,
                      stride=1,
                      padding=1 // 2,
                      has_bias=False,pad_mode=padding_mode), nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels , in_channels, 1, has_bias=False,pad_mode=padding_mode),
            nn.Sigmoid()
            )
        self.temporal_conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=padding,dilation=dilation,group=groups,has_bias=bias,pad_mode=padding_mode)

        self.spatial_conv=SpatialConv(dim_in=in_channels,dim_out=out_channels,pos_dim=pos_dim)
    def construct(self,x):
        param=self.param_conv(x)
        x=self.temporal_conv(param*x)+self.spatial_conv(x,param)
        return x



def conv3x3x3(in_planes, out_planes, stride=1,pos_dim=7):

    return Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False,pos_dim=pos_dim)


def conv1x1x1(in_planes, out_planes, stride=1):
    
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     has_bias=False,pad_mode='pad')


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 =conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 =conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,shortcut_conv=None,pos_dim=7):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3x3(planes, planes,pos_dim=pos_dim)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        # self.downsample = downsample
        self.shortcut_conv=shortcut_conv
        self.stride = stride
        if stride !=1:
            self.downsample= nn.SequentialCell( nn.Conv2d(in_planes,in_planes,kernel_size=2,stride=2,group=in_planes,has_bias=True,pad_mode='pad'), nn.BatchNorm2d(in_planes))
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def construct(self, x):
        if self.stride!=1:
            x=self.downsample(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut_conv is not None:
            residual=self.shortcut_conv(x)

        out += residual
        out = self.relu(out)

        return out



class SQTNet_base(nn.Cell):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,dropout=0.2,spatial_stride=[1,2,2,2],pos_dim=[56,28,14,7]):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.dropout =dropout
        self.conv1 = nn.Conv2d(n_input_channels,
                               self.in_planes,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               group=1,
                               has_bias=False,pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()
        self.maxpool = nn.SequentialCell([
               nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT"),
               nn.MaxPool2d(kernel_size=3, stride=2)]) #nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same') #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type,stride=spatial_stride[0],pos_dim=pos_dim[0])
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=spatial_stride[1],pos_dim=pos_dim[1])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=spatial_stride[2],pos_dim=pos_dim[2])
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=spatial_stride[3],pos_dim=pos_dim[3])

        self.avgpool = nn.AdaptiveAvgPool2d(( 1, 1))
        self.fc = nn.Dense(block_inplanes[3] * block.expansion, n_classes) #nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.activation=nn.Softmax(axis=-1)
        self.dropout_layer=nn.Dropout(keep_prob=self.dropout)
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d,nn.Conv3d)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
                cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, (nn.Dense)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                    cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1,pos_dim=7):
        shortcut=None
        if  self.in_planes != planes * block.expansion:
            shortcut = nn.SequentialCell(
                conv1x1x1(self.in_planes, planes * block.expansion, stride=1),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,shortcut_conv=shortcut,pos_dim=pos_dim))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes,pos_dim=pos_dim))

        return nn.SequentialCell(*layers)

    def construct(self, x): 
        if isinstance(x, dict):
            x = x["video"]
        N,C,T,H,W=x.shape
        x=x.view(N,-1,H,W)                                                                                                                                                   
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.shape[0], -1)
        logits=x
        if self.dropout > 0.:
            x = self.dropout_layer(x)
        x = self.fc(x)
        if not self.training:
            x = self.activation(x)

        return x
    def extract_features(self,x):
        N,C,T,H,W=x.shape
        x=x.view(N,-1,H,W)                                                                                                                                                   
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)#t16, N,1024,6,6
        x4 = self.layer4(x3)#t8, N,2048,3,3
        # x3=x3.
        return [x3,x4]

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = SQTNet_base(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = SQTNet_base(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = SQTNet_base(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = SQTNet_base(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = SQTNet_base(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = SQTNet_base(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = SQTNet_base(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

@MODELS.register_module()
class SqueezeTime(nn.Cell):
    def __init__(self,depth=50,widen_factor=1.0,dropout=0.5,input_channels=48,n_classes=400,load=None,spatial_stride=[1,2,2,2],pos_dim=[64,32,16,8]):
        super(SqueezeTime,self).__init__()
        model=generate_model(depth,widen_factor=widen_factor,dropout=dropout,n_input_channels=input_channels,n_classes=n_classes,spatial_stride=spatial_stride,pos_dim=pos_dim)
        if load is not None:
            print('Load {} into the backbone model...'.format(load))
            ms.load_checkpoint(load, model)
            print('Load pretrained model Done!')
        self.net=model
    def construct(self,x):
        return self.net(x)

def get_SQTNet(n_input_channels=48,pretrained_path='TAL/SQTNet_71.64_ms.ckpt',widen_factor=1.0):
    model =generate_model(50,widen_factor=widen_factor, dropout=0.5,n_input_channels=n_input_channels,n_classes=400)
    print('Load {} into the backbone model...'.format(pretrained_path))
    # ms.load_checkpoint(pretrained_path, model,filter_prefix='conv1')
    ms.load_checkpoint(pretrained_path, model)
    print('Load pretrained model Done!')
    return model