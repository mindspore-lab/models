# from dataloader import *
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
import sys
import argparse
from mindspore import context
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
# context.set_context(mode=context.PYNATIVE_MODE,device_id=0, device_target="GPU")
context.set_context(mode=context.GRAPH_MODE,device_id=0, device_target="GPU")


def denselayer(inputs, output_size):
    fc = nn.Dense(inputs, output_size)
    return fc

def lrelu(alphas):
    return nn.LeakyReLU(alpha=alphas)


def batchnorm(inputs):
    batchn = nn.BatchNorm2d(inputs, eps=0.001)
    return batchn

def conv2(batch_input, kernel=3, output_channels=64, stride=1, use_bias=True):
    padding = int((kernel - 1) / 2)
    if use_bias:
        conv = nn.Conv2d(in_channels=batch_input, out_channels=output_channels, kernel_size=kernel, stride=stride, pad_mode='pad',padding=padding, has_bias=True)
    else:
        conv = nn.Conv2d(in_channels=batch_input, out_channels=output_channels, kernel_size=kernel, stride=stride,pad_mode='pad',padding=padding, has_bias=False)
    return conv

def discriminator_block(inputs,output_channel,kernel_size,stride):
    conv2d=conv2(inputs,kernel_size,output_channel,stride,use_bias=False)
    batchn=nn.BatchNorm2d(output_channel,eps=0.001)
    leaky_relu = nn.LeakyReLU(alpha=0.2)
    net=nn.SequentialCell(conv2d,batchn,leaky_relu)
    return net


def residual_block(inputs, output_channel=64, stride=1):
    net = nn.SequentialCell(conv2(inputs, 3, output_channel, stride, use_bias=True), nn.ReLU(),
                        conv2(output_channel, 3, output_channel, stride, use_bias=False))

    return net


class discriminator_pic(nn.Cell):
    def __init__(self,args=None):
        super(discriminator_pic, self).__init__()
        if args is None:
            raise ValueError("No args is provided for discriminator")
        self.conv = nn.SequentialCell(conv2(1, 3, 64, 1), lrelu(0.2))
        self.block1 = discriminator_block(64, 64, 4, 2)
        self.resids1 = nn.CellList(
            [nn.SequentialCell(residual_block(64, 64, 1), batchnorm(64)) for i in range(int(args.discrim_resblocks))])
        self.block2 = discriminator_block(64, args.discrim_channels, 4, 2)
        self.resids2 = nn.CellList([nn.SequentialCell(residual_block(args.discrim_channels, args.discrim_channels, 1),
                                                    batchnorm(args.discrim_channels)) for i in
                                      range(int(args.discrim_resblocks))])
        self.block3 = discriminator_block(args.discrim_channels, args.discrim_channels, 4, 2)
        self.resids3 = nn.CellList([nn.SequentialCell(residual_block(args.discrim_channels, args.discrim_channels, 1),
                                                    batchnorm(args.discrim_channels)) for i in
                                      range(int(args.discrim_resblocks))])
        self.block4 = discriminator_block(args.discrim_channels, 64, 4, 2)
        self.block5 = discriminator_block(64, 9, 4, 2)
        self.fc = denselayer(2304, 1)

    def construct(self,x):
        layer_list = []
        net = self.conv(x)
        net=self.block1(net)
        for block in self.resids1:
            net=block(net)+net
        layer_list.append(net)
        net=self.block2(net)
        for block in self.resids2:
            net = block(net) + net
        net = self.block3(net)
        layer_list.append(net)
        for block in self.resids3:
            net=block(net)+net
        layer_list.append(net)
        net=self.block4(net)
        layer_list.append(net)
        net = self.block5(net)
        net=net.view(net.shape[0], -1)
        net=self.fc(net)
        net=mindspore.ops.sigmoid(net)
        return net, layer_list

class FNet(nn.Cell):
    def __init__(self,in_nc):
        super(FNet,self).__init__()
        self.encoder1=nn.SequentialCell(
            nn.Conv2d(2*in_nc,32,3,1,pad_mode='pad',padding=1,has_bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.SequentialCell(
            nn.Conv2d(32, 64, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.SequentialCell(
            nn.Conv2d(64, 128, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.decoder1 = nn.SequentialCell(
            nn.Conv2d(128, 256, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
        )
        self.decoder2 = nn.SequentialCell(
            nn.Conv2d(256, 128, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
        )
        self.decoder3 = nn.SequentialCell(
            nn.Conv2d(128, 64, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
        )
        self.flow=nn.SequentialCell(
            nn.Conv2d(64, 32, 3, 1, pad_mode='pad',padding=1, has_bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, 3, 1,pad_mode='pad',padding=1, has_bias=True),
        )

    def construct(self, x1,x2):
        out = self.encoder1(mindspore.ops.cat([x1, x2], axis=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = mindspore.ops.interpolate(
            self.decoder1(out), size=(128,128), mode='bilinear', align_corners=False)
        out = mindspore.ops.interpolate(
            self.decoder2(out), size=(256,256), mode='bilinear', align_corners=False)
        out = mindspore.ops.interpolate(
            self.decoder3(out), size=(512,512), mode='bilinear', align_corners=False)
        out = mindspore.ops.tanh(self.flow(out)) * 24
        return out



def backward_warp(x, flow, mode='bilinear', padding_mode='border'):

    n, c, h, w = x.shape

    # create mesh grid
    iu = mindspore.ops.linspace(-1.0, 1.0, w).view(1, 1, 1, w).broadcast_to((n, -1, h, -1))
    iv = mindspore.ops.linspace(-1.0, 1.0, h).view(1, 1, h, 1).broadcast_to((n, -1, -1, w))
    grid = mindspore.ops.cat([iu, iv], 1)
    flow = mindspore.ops.cat([
        flow[:, 0:1, ...] / ((w - 1.0) / 2.0),
        flow[:, 1:2, ...] / ((h - 1.0) / 2.0)], axis=1)

    # add flow to grid and reshape to nhw2
    grid = (grid + flow).permute(0, 2, 3, 1)
    output = mindspore.ops.grid_sample(x, grid, mode=mode, padding_mode=padding_mode,align_corners=True)

    return output


def space_to_depth(x, scale):
    """ Equivalent to tf.space_to_depth()
    """

    n, c, in_h, in_w = x.size()
    out_h, out_w = in_h // scale, in_w // scale

    x_reshaped = x.reshape(n, c, out_h, scale, out_w, scale)
    x_reshaped = x_reshaped.permute(0, 3, 5, 1, 2, 4)
    output = x_reshaped.reshape(n, scale * scale * c, out_h, out_w)

    return output



class generator(nn.Cell):
    def __init__(self,gen_output_channels,args=None):
        super(generator, self).__init__()
        if args is None:
            raise ValueError("No args is provided for generator")
        self.num=args.num_resblock
        self.block_1_1=None
        self.block_2_1 = None
        self.block_3_1 = None
        self.block_4_1 = None
        self.block_5 = None
        self.block_4_2 = None
        self.block_3_2 = None
        self.block_2_2 = None
        self.block_1_2 = None
        self.create_model()

    @staticmethod
    def add_block_conv(in_channels,out_channels, kernel_size, stride,padding,batchOn,ReluOn):
        seq=[]
        conv=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                       stride=stride, pad_mode='pad',padding=padding, has_bias=True)
        seq.append(conv)
        if batchOn:
            batch_norm=batchnorm(out_channels)
            seq.append(batch_norm)
        if ReluOn:
            seq.append((nn.ReLU()))
        return seq

    @staticmethod
    def add_block_conv_transpose(in_channels, out_channels, kernel_size, stride, padding, batchOn,
                                 ReluOn):
        seq=[]
        convt=nn.Conv2dTranspose(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                 stride=stride,pad_mode='pad',padding=padding)
        seq.append(convt)
        if batchOn:
            batch_norm=batchnorm(out_channels)
            seq.append(batch_norm)
        if ReluOn:
            seq.append((nn.ReLU()))
        return seq


    def create_model(self):
        kernel_size=3
        padding=kernel_size//2
        block_1_1=[]
        block_1_1.extend(self.add_block_conv(2,64,kernel_size,1,padding,True,True))
        block_1_1.extend(self.add_block_conv(64, 64, kernel_size, 1, padding, True, True))
        block_1_1.extend(self.add_block_conv(64, 64, kernel_size, 1, padding, True, True))
        self.block_1_1=nn.SequentialCell(*block_1_1)

        block_2_1 = [nn.MaxPool2d(kernel_size=2,stride=2)]
        block_2_1.extend(self.add_block_conv(64, 128, kernel_size, 1, padding, True, True))
        block_2_1.extend(self.add_block_conv(128, 128, kernel_size, 1, padding, True, True))
        self.block_2_1=nn.SequentialCell(*block_2_1)

        block_3_1 = [nn.MaxPool2d(kernel_size=2,stride=2)]
        block_3_1.extend(self.add_block_conv(128, 256, kernel_size, 1, padding, True, True))
        block_3_1.extend(self.add_block_conv(256, 256, kernel_size, 1, padding, True, True))
        self.block_3_1 = nn.SequentialCell(*block_3_1)

        block_4_1 = [nn.MaxPool2d(kernel_size=2,stride=2)]
        block_4_1.extend(self.add_block_conv(256, 512, kernel_size, 1, padding, True, True))
        block_4_1.extend(self.add_block_conv(512, 512, kernel_size, 1, padding, True, True))
        self.block_4_1 = nn.SequentialCell(*block_4_1)

        block_5_1 = [nn.MaxPool2d(kernel_size=2,stride=2)]
        block_5_1.extend(self.add_block_conv(512, 1024, kernel_size, 1, padding, True, True))
        block_5_1.extend(self.add_block_conv(1024, 1024, kernel_size, 1, padding, True, True))
        block_5_1.extend(self.add_block_conv_transpose(1024,512,kernel_size+1, 2, padding, True, True))
        self.block_5 = nn.SequentialCell(*block_5_1)

        block_4_2 = []
        block_4_2.extend(self.add_block_conv(1024, 512, kernel_size, 1, padding, True, True))
        block_4_2.extend(self.add_block_conv(512, 512, kernel_size, 1, padding, True, True))
        block_4_2.extend(self.add_block_conv_transpose(512, 256, kernel_size+1, 2, padding, True, True))
        self.block_4_2=nn.SequentialCell(block_4_2)

        block_3_2 = []
        block_3_2.extend(self.add_block_conv(512, 256, kernel_size, 1, padding, True, True))
        block_3_2.extend(self.add_block_conv(256, 256, kernel_size, 1, padding, True, True))
        block_3_2.extend(self.add_block_conv_transpose(256, 128, kernel_size+1, 2, padding, True, True))
        self.block_3_2 = nn.SequentialCell(block_3_2)

        block_2_2 = []
        block_2_2.extend(self.add_block_conv(256, 128, kernel_size, 1, padding, True, True))
        block_2_2.extend(self.add_block_conv(128, 128, kernel_size, 1, padding, True, True))
        block_2_2.extend(self.add_block_conv_transpose(128, 64, kernel_size+1, 2, padding, True, True))
        self.block_2_2 = nn.SequentialCell(block_2_2)

        block_1_2 = []
        block_1_2.extend(self.add_block_conv(128, 64, kernel_size, 1, padding, True, True))
        block_1_2.extend(self.add_block_conv(64, 64, kernel_size, 1, padding, True, True))
        block_1_2.extend(self.add_block_conv(64, 1, 1, 1, 0, False, False))
        self.block_1_2 = nn.SequentialCell(block_1_2)

    def construct(self, input):
        block_1_1_output=self.block_1_1(input)
        block_2_1_output=self.block_2_1(block_1_1_output)
        block_3_1_output=self.block_3_1(block_2_1_output)
        block_4_1_output=self.block_4_1(block_3_1_output)
        block_5_output=self.block_5(block_4_1_output)
        # block_5_output=(block_5_output)
        # result2 = mindspore.ops.cat((block_4_1_output, block_2_1_output), axis=1)
        result=self.block_4_2(mindspore.ops.cat((block_4_1_output,block_5_output),axis=1))
        result=self.block_3_2(mindspore.ops.cat((block_3_1_output,result),axis=1))
        result = self.block_2_2(mindspore.ops.cat((block_2_1_output, result), axis=1))
        result = self.block_1_2(mindspore.ops.cat((block_1_1_output, result), axis=1))
        result=result+input[:,0,:,:].unsqueeze(1)
        return result







#



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--discrim_resblocks',type=int,default=4)
    parser.add_argument('--discrim_channels',type=int,default=128)
    parser.add_argument('--num_resblock', type=int, default=128)
    args=parser.parse_args()
    generator_F=generator(1,args=args)
    x=mindspore.ops.rand((4,2,512,512))
    generator_F(x)
    print('t')

